import torch
import time


class CodeAttack:
    def __init__(self,
                 model,
                 tokenizer,
                 num_starts=1,
                 num_steps=5000,
                 learning_rate=10,
                 momentum=0.9,
                 gather_grad=False):

        self.model = model
        self.tokenizer = tokenizer
        self.num_starts = num_starts
        self.num_steps = num_steps

        self.lr = learning_rate
        self.momentum = momentum

        self.device = model.device
        self.dtype = model.dtype

        self.embed_mat = model.model.embed_tokens.weight.float()
        self.vocal_size = self.embed_mat.shape[0]

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.buffer_size = 20

        # sparsity setting
        self.min_sparsity = 2
        self.warmup_steps = 100
        self.low_loss_thres = 0.5
        self.illegal_tokens = tuple(tokenizer.added_tokens_decoder.keys())

        # search setting
        self.topK = 32
        self.bs = 64

        self.gather_grad = gather_grad
        self.gather_grad &= torch.distributed.is_initialized()

        if self.gather_grad:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    def avg_ddp(self, tensor):
        # avergae the tensor across ranks when ddp initialized
        if self.gather_grad:
            tensor = tensor.to(self.device)
            torch.distributed.all_reduce(tensor)
            tensor /= self.world_size
        return tensor

    def check_legal_input(self, multi_examples):

        num_adv_tokens = None

        for example in multi_examples:
            tokens = example['tokens'].view(1, -1).to(self.device)
            example['tokens'] = tokens

            suffix_slice = example['slices']['adv_slice']
            num_adv_tokens = suffix_slice.stop - suffix_slice.start

        self.num_adv_tokens = num_adv_tokens
        return multi_examples

    @torch.no_grad()
    def prepare_stuffs(self, multi_examples):
        self.check_legal_input(multi_examples)

        self.gt_label_list, self.logit_slice_list = [], []

        for example in multi_examples:
            tokens = example['tokens']
            target_slice = example["slices"]['target_slice']

            target_start, target_stop = target_slice.start, target_slice.stop

            gt_label = tokens[:, target_slice]
            gt_label = gt_label.expand(self.buffer_size, -1)

            logit_slice = slice(target_start - 1, target_stop - 1)

            self.gt_label_list.append(gt_label)
            self.logit_slice_list.append(logit_slice)

        self.num_examples = len(multi_examples)

    def get_optimizer(self, num_adv_tokens):
        soft_opt = torch.randn(
            size=[self.num_starts, num_adv_tokens, self.vocal_size]
        ).to(self.device)

        if self.gather_grad:
            torch.distributed.all_reduce(soft_opt)
            soft_opt /= self.world_size**.5

        soft_opt = soft_opt.softmax(dim=2)
        soft_opt.requires_grad = True
        return soft_opt

    @torch.no_grad()
    def make_sparse(self, soft_opt, step_, n_onehot):
        assert len(soft_opt.shape) == 2
        if step_ <= self.warmup_steps:
            r = step_ / self.warmup_steps
            K = 1024 * (self.min_sparsity / 1024) ** r
            K = round(K)
        else:
            K = self.min_sparsity

        point = soft_opt.detach().clone()
        point[..., self.illegal_tokens] = -1000
        mask = point >= point.topk(K, dim=-1)[0][..., -1:]
        point = torch.where(mask, point.relu() + 1e-6, 0)
        point /= point.sum(dim=-1, keepdim=True)

        if step_ > self.warmup_steps and n_onehot > 1:

            # make sure token_idxes is the same on different GPUs
            token_idxes = torch.randn(self.num_adv_tokens)
            token_idxes = token_idxes.to(self.device)
            token_idxes = self.avg_ddp(token_idxes)
            token_idxes = token_idxes.topk(n_onehot)[1]

            for token_idx in token_idxes:
                vec = point[token_idx]
                vec_argmax = vec.argmax()
                point[token_idx] = 0
                point[token_idx, vec_argmax] = 1
        return point

    @torch.no_grad()
    def evaluate(self, buffer_set, multi_examples):
        adv_tokens = list(buffer_set)
        if len(adv_tokens) < self.buffer_size:
            adv_tokens += adv_tokens[:1] * (self.buffer_size - len(adv_tokens))
        adv_tokens = torch.tensor(adv_tokens,
                                  dtype=torch.int64,
                                  device=self.device)

        avg_loss, avg_acc = 0, 0
        for exmaple_idx in range(self.num_examples):
            example = multi_examples[exmaple_idx]
            full_samples = self.update_inputids(example, adv_new_tokens=adv_tokens)

            outputs = self.model(input_ids=full_samples)

            outputs = outputs.logits[:, self.logit_slice_list[exmaple_idx]]
            pred = outputs.argmax(dim=-1)
            target_slice = example['slices']['target_slice']
            _label = full_samples[:, target_slice]

            losses = self.loss_fn(outputs.mT, _label)

            acc_per_example = pred.eq(_label).float().mean(1)
            avg_acc += acc_per_example

            loss_per_example = losses.pow(2).mean(1).sqrt()
            avg_loss += loss_per_example

        avg_loss /= self.num_examples
        avg_acc /= self.num_examples

        avg_loss = self.avg_ddp(avg_loss)
        avg_acc = self.avg_ddp(avg_acc)

        best_acc = avg_acc.max().item()

        best_loss = avg_loss.min().item()

        best_adv = adv_tokens[avg_loss.argmin()]

        if best_acc == 1:
            good_adv = adv_tokens[avg_acc.argmax()]

            return best_acc, best_loss, good_adv, True
        if best_acc >= 0.98:
            print(pred.eq(_label)[0])
        return best_acc, best_loss, best_adv, False

    def clean_cache(self):
        self.num_adv_tokens = None
        self.gt_label_list = None
        self.logit_slice_list = None
        self.num_examples = None
        torch.cuda.empty_cache()

    def update_inputids(self, example, adv_new_tokens):
        bs = adv_new_tokens.shape[0] if len(adv_new_tokens.shape)>1 else 1
        input_ids = example['tokens'][0].expand(bs, -1).clone()
        slices = example['slices']
        adv_new_tokens = torch.tensor(adv_new_tokens).to(input_ids.device)
        if 'target_adv_slice' in slices:
            input_ids[:, slices['target_adv_slice']] = adv_new_tokens
        if 'adv_slice' in slices:
            input_ids[:, slices['adv_slice']] = adv_new_tokens

        if 'history_slice' in slices:
            for hist in slices['history_slice']:
                input_ids[:, hist] = adv_new_tokens
        return input_ids

    def prepare_embeds(self, example, soft_opt):
        bs = soft_opt.shape[0] if len(soft_opt.shape)>2 else 1

        tokens = example['tokens']
        slices = example['slices']
        embeds = self.model.model.embed_tokens(tokens).detach()

        adv_embeds = soft_opt @ self.embed_mat
        soft_opt_onehot = soft_opt.max(dim=-1, keepdim=True)[0] == soft_opt
        adv_onehot_embeds = soft_opt_onehot.float() @ self.embed_mat
        adv_embeds = adv_embeds.to(self.dtype)

        full_embeds = embeds.expand(bs, -1, -1).clone()
        one_hot_opt = soft_opt == soft_opt.max(dim=-1, keepdim=True)[0]

        if 'target_adv_slice' in slices:
            full_embeds[:, slices['target_adv_slice']] = one_hot_opt.float() @ self.embed_mat
        if 'adv_slice' in slices:
            full_embeds[:, slices['adv_slice']] = adv_embeds

        if 'history_slice' in slices:
            for hist in slices['history_slice']:
                full_embeds[:, hist] = adv_embeds

        return full_embeds


    def get_resample_ids(self):
        lookup = set()
        token_idx = torch.randint(self.num_adv_tokens, size=[self.bs * 2])
        cand_idx = torch.randint(self.topK, size=[self.bs * 2])
        for i, j in zip(token_idx, cand_idx):
            lookup.add((i.item(), j.item()))
            if len(lookup) == self.bs:
                return torch.tensor(list(lookup))

        while len(lookup) < self.bs:
            i = torch.randint(self.num_adv_tokens, size=[1])
            j = torch.randint(self.topK, size=[1])
            lookup.add((i.item(), j.item()))

        lookup = list(lookup)
        lookup = torch.tensor(lookup)
        return lookup

    def attack(self, multi_examples):
        self.prepare_stuffs(multi_examples)
        soft_opt = self.get_optimizer(self.num_adv_tokens)
        seen_set, buffer_set = set(), set()
        onehot_loss, onehot_acc, final_adv = 1000, 0, None

        momentum_buffer, low_loss_count = 0, 0
        start_time = time.time()
        for step_ in range(self.num_steps + 1):
            soft_opt.requires_grad = True
            soft_opt.grad = None

            total_ell = 0
            for exmaple_idx in range(self.num_examples):

                example = multi_examples[exmaple_idx]
                full_embeds = self.prepare_embeds(example, soft_opt)

                outputs = self.model(inputs_embeds=full_embeds)
                logits = outputs.logits[:, self.logit_slice_list[exmaple_idx]]

                _tokens = example['tokens'].expand(soft_opt.shape[0], -1).clone()
                if "target_adv_slice" in example['slices']:
                    _tokens[:, example['slices']['target_adv_slice']] = soft_opt.argmax(-1)
                _label = _tokens[:, example['slices']['target_slice']]
                loss_per_sample = self.loss_fn(logits.mT, _label)

                loss_per_sample = loss_per_sample.pow(2).mean(dim=1).sqrt()

                ell = loss_per_sample.mean()
                total_ell = total_ell + ell

            total_ell = total_ell / self.num_examples

            total_ell.backward()
            total_ell = self.avg_ddp(total_ell)

            speed = (time.time() - start_time) / (step_ + 1)

            if total_ell < 0.5:
                low_loss_count += 1
            else:
                low_loss_count -= 0.01
                low_loss_count = max(low_loss_count, 0)

            n_onehot = min(round(low_loss_count ** .5), self.num_adv_tokens - 2)
            print(f'Step: {step_}, loss: {total_ell.item(): .3f}, n_onehot: {n_onehot}, speed: {speed: .2f} sec/iter')
            grad = soft_opt.grad.data.clone()[0]
            grad = self.avg_ddp(grad)

            momentum_buffer = momentum_buffer * self.momentum + grad
            momentum_buffer[..., self.illegal_tokens] = 10 ** 10
            candidates = momentum_buffer.topk(k=self.topK, dim=1, largest=False)[1]

            # sample new update entries
            if self.gather_grad:
                if self.rank == 0:
                    resample_ids = self.get_resample_ids().float() * self.world_size
                else:
                    resample_ids = torch.zeros(self.bs, 2).to(self.device)

                resample_ids = self.avg_ddp(resample_ids).long().tolist()
            else:
                resample_ids = self.get_resample_ids()

            # find the best new entry
            new_points = []
            for (i, j) in resample_ids:
                new_update = torch.zeros_like(momentum_buffer)
                k = candidates[i, j]
                new_update[i, k] = momentum_buffer[i, k].item()
                new_point = soft_opt.data.clone()[0]
                new_point -= self.lr * new_update

                non_sparse_point = new_point.clone()
                non_sparse_point[..., self.illegal_tokens] = -1000

                new_point = self.make_sparse(new_point, step_, n_onehot)
                new_points += [(new_point, non_sparse_point)]

            all_points = torch.stack([i[0] for i in new_points])

            tmp_loss = 0
            for exmaple_idx in range(self.num_examples):

                example = multi_examples[exmaple_idx]
                full_embeds = self.prepare_embeds(example, all_points)
                outputs = self.model(inputs_embeds=full_embeds)
                logits = outputs.logits[:, self.logit_slice_list[exmaple_idx]]

                _tokens = example['tokens'].expand(all_points.shape[0], -1).clone()
                if "target_adv_slice" in example['slices']:
                    _tokens[:, example['slices']['target_adv_slice']] = all_points.argmax(-1)
                _label = _tokens[:, example['slices']['target_slice']]
                loss_per_sample = self.loss_fn(logits.mT, _label)

                loss_per_sample = loss_per_sample.pow(2).mean(dim=1).sqrt()

                tmp_loss = tmp_loss + loss_per_sample

            tmp_loss = self.avg_ddp(tmp_loss)
            best_idx = tmp_loss.argmin()
            new_point, non_sparse_point = new_points[best_idx]
            soft_opt.data.copy_(new_point)

            # one hot evaluation
            adv_token = non_sparse_point.argmax(dim=1)
            adv_token = tuple(adv_token.tolist())

            if adv_token not in seen_set and len(adv_token) == self.num_adv_tokens:
                seen_set.add(adv_token)
                buffer_set.add(adv_token)
            else:
                for i in range(self.num_adv_tokens):
                    adv_token1 = list(adv_token)
                    adv_token1[i] = non_sparse_point[i].topk(2)[1][1].item()
                    adv_token1 = tuple(adv_token1)
                    if adv_token1 not in seen_set and len(adv_token1) == self.num_adv_tokens:
                        seen_set.add(adv_token1)
                        buffer_set.add(adv_token1)
                        break

            if len(buffer_set) == self.buffer_size:

                out = self.evaluate(buffer_set, multi_examples)

                batch_acc, batch_loss, best_adv, early_stop = out

                if batch_loss < onehot_loss:
                    onehot_loss = batch_loss
                    final_adv = best_adv
                    if batch_loss < 0.5:
                        print(final_adv)

                onehot_acc = max(onehot_acc, batch_acc)

                print(f'iter:{step_}, '
                      f'loss_batch:{total_ell: .2f}, '
                      f'best_loss:{onehot_loss: .2f}, '
                      f'best_acc:{onehot_acc: .2f}')
                if early_stop:
                    print('Early Stop with an Exact Match!')
                    self.clean_cache()
                    return onehot_loss, best_adv, step_, early_stop
                buffer_set = set()

        if len(buffer_set) > 0:

            out = self.evaluate(buffer_set, multi_examples)

            batch_acc, batch_loss, best_adv, early_stop = out

            if batch_loss < onehot_loss:
                onehot_loss = batch_loss
                final_adv = best_adv

            onehot_acc = max(onehot_acc, batch_acc)

            print(f'iter:{step_}, '
                  f'loss_batch:{total_ell: .2f}, '
                  f'best_loss:{onehot_loss: .2f}, '
                  f'best_acc:{onehot_acc: .2f}')
            if early_stop:
                print('Early Stop with an Exact Match!')
                self.clean_cache()
                return onehot_loss, best_adv, step_, early_stop

        self.clean_cache()
        return onehot_loss, final_adv, step_, early_stop
