import string
import torch
from pyro.ops.contract import einsum

bsz = 16
num_ent = 7

dims = (bsz,) + (2,) * num_ent

# bsz x 2 x 2 x ... where there are num_ent 2s
full_potentials = torch.zeros(dims)

unaries = torch.randn(bsz, num_ent, 2)
binary_potentials = torch.randn(bsz, num_ent*(num_ent-1)//2, 2,2)

ix = 0
assert unaries.size(1) == num_ent
for i in range(num_ent):
    this_unaries = unaries[:,i]
    assert this_unaries.size() == (bsz, 2)

    # make this_unaries broadcastable with full_potentials
    # TODO: ther emust be some better way to do this
    for d in range(1, num_ent+1):
        if d == i + 1:
            continue
        this_unaries = this_unaries.unsqueeze(d)

    full_potentials += this_unaries

    # iterate over all pairs that include dot i that we haven't yet seen
    for j in range(num_ent):
        if i >= j:
            continue
        # make binary_potentials broadcastable with full_potentials
        this_binary = binary_potentials[:,ix]
        # add one to i and j for the batch dimension
        dims_to_unsqueeze = (d for d in range(1, num_ent+1) if d != (i+1) and d != (j+1))
        for d in dims_to_unsqueeze:
            this_binary = this_binary.unsqueeze(d)

        full_potentials += this_binary

        ix += 1

assert ix == binary_potentials.size(1)

# probs = full_potentials.view(full_potentials.size(0), -1).softmax(-1).view_as(full_potentials)

# pyro version

var_names = list(string.ascii_lowercase[:num_ent])

batch_name='z'
assert batch_name not in var_names

unary_factor_names = []
unary_factors = []

binary_factor_names = []
binary_factors = []

ix = 0
for i in range(num_ent):
    unary_factor_names.append(batch_name + var_names[i])
    unary_factors.append(unaries[:,i])
    for j in range(num_ent):
        if i >= j:
            continue
        binary_factor_names.append(batch_name + var_names[i] + var_names[j])
        binary_factors.append(binary_potentials[:,ix])

        ix += 1
assert ix == binary_potentials.size(1)

contraction_string = '{}->z{}'.format(','.join(unary_factor_names + binary_factor_names), ''.join(var_names))
full_potentials_pyro, = einsum(contraction_string, *unary_factors, *binary_factors, modulo_total=True, plates='', backend='pyro.ops.einsum.torch_log')

#assert torch.allclose(full_potentials, full_potentials_pyro)

probs = full_potentials_pyro.contiguous().view(bsz, -1).softmax(-1).view_as(full_potentials_pyro)

marginalized = []
for i in range(num_ent):
    d = i + 1
    this_marg = probs
    for d in range(1, num_ent+1):
        if d == i+1:
            continue
        this_marg = this_marg.sum(d, keepdim=True)
    this_marg = this_marg.view(bsz, -1)
    marginalized.append(this_marg.log())

output_factor_names = ','.join('z{}'.format(v) for v in var_names)

marginalized_pyro = [t.log() for t in einsum('z{}->{}'.format(''.join(var_names), output_factor_names), probs, modulo_total=True)]

unary_factors = unaries.transpose(0,1)
binary_factors = binary_potentials.transpose(0,1)

marginalized_pyro_oneshot = [t.log_softmax(-1) for t in einsum(
    '{}->{}'.format(','.join(unary_factor_names + binary_factor_names), output_factor_names),
    *unary_factors, *binary_factors,
    modulo_total=True,
    #plates='',
    backend='pyro.ops.einsum.torch_log'
)]

# for x, y, z in zip(marginalized, marginalized_pyro, marginalized_pyro_oneshot):
#     assert torch.allclose(x, y) and torch.allclose(y, z)
