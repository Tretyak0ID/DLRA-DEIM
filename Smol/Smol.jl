export gen_tensor_rhs, gen_initial_conditions, calc_rhs, gen_delta_mono_source_tensor, gen_zero_initial_conditions, gen_qtt_rhs

function gen_tensor_rhs(N, d; alpha = 0.0, beta = 0.0)
	I1 = [if i + j == k 1 else 0 end for i = 1 : N, j = 1 : N, k = 1 : N]
	I2 = [if i == k 1 else 0 end     for i = 1 : N, j = 1 : N, k = 1 : N]
	I3 = [if j == k 1 else 0 end     for i = 1 : N, j = 1 : N, k = 1 : N]

	K = [i.^alpha.*j.^beta + j.^alpha.*i.^beta for i = 1 : N, j = 1 : N, k = 1 : N]

	S = reshape(1/2*K.*(I1 - I2 - I3), ntuple(i->2,3*d)...)
	S = permutedims(S,  [i + (j - 1)*d for i in 1:d for j in 1:3])
	S = reshape(S, ntuple(i->2^3, d)...)
	S = TTsvd(S; tol_rel = 1e-4)
end

function gen_qtt_rhs(N, d; alpha = 0.0, beta = 0.0)
	coresI1 = CoreCell(undef, d)
	# coresI2 = CoreCell(undef, d)
	# coresI3 = CoreCell(undef, d)
	# coresK  = CoreCell(undef, d)
	# [coresI2[k] = reshape([1, 0, 1, 0, 0, 1, 0, 1], (1, 8, 1)) for k = 1 : d]
	# [coresI3[k] = reshape([1, 0, 0, 1, 1, 0, 0, 1], (1, 8, 1)) for k = 1 : d]
	coresI1[1] = reshape([Int(i + j == k + 2*beta) for i in 0 : 1 for j in 0 : 1 for k in 0 : 1 for beta in 0 : 1], (1, 8 ,2))
	#[coresI1[m] = reshape([Int(i + j + alpha == k + 2*beta) for alpha in 0 : 1 for i in 0 : 1 for j in 0 : 1 for k in 0 : 1 for beta in 0 : 1], (2, 8, 2)) for m = 2 : d - 1]
	coresI1[d] = reshape([Int(i + j + alpha == k) for alpha in 0 : 1 for i in 0 : 1 for j in 0 : 1 for k in 0 : 1], (2, 8, 1))

	# Kten = [i.^alpha.*j.^beta + j.^alpha.*i.^beta for i = 1 : N, j = 1 : N]
	# Kten = reshape(Kten, ntuple(i->2,2*d)...)
	# Kten = permutedims(Kten,  [i + (j - 1)*d for i in 1:d for j in 1:2])
	# Kten = reshape(Kten, ntuple(i->2^2, d)...)
	# Kten = TTsvd(Kten; tol_rel = 1e-4)
	# [coresK[k] = reshape(permutedims(reshape(Kten.cores[k], size(Kten.cores[k])..., 1) .* ones(1, 1, 1, 2), [1,2,4,3]), (size(Kten.cores[k],1), 2^3, size(Kten.cores[k],3))) for k = 1 : d]
	
	I1 = TTtensor(coresI1)
	# I2 = TTtensor(coresI2)
	# I3 = TTtensor(coresI3)
	# K  = TTtensor(coresK)

	#return 1/2*ewprod(K, (I1 - I2 - I3))
	I1
end

function gen_initial_conditions(N, d)
	c = zeros(N,1);
	c[1] = 1;
	c = reshape(c, ntuple(i->2, d))
end

function gen_zero_initial_conditions(N, d)
	c = zeros(N,1);
	c = reshape(c, ntuple(i->2, d))
end

function gen_ones_initial_conditions(N, d)
	c = rand(N,1);
	c = reshape(c, ntuple(i->2, d))
end

function gen_delta_mono_source_tensor(N, d)
	c = zeros(N,1);
	c[1] = 1;
	c = reshape(c, ntuple(i->2, d))
end

function calc_rhs(ctt, Stt)
	S_cores = Stt.cores
	c_cores = ctt.cores
	dn 		= size(S_cores, 1)
	cells = CoreCell(undef, dn)
		
	for k = 1 : dn
		coreS = S_cores[k]
    	corec = c_cores[k]
		
		rk_1 = size(coreS, 1)
        rk   = size(coreS, 3)
        Rk_1 = size(corec, 1)
        Rk   = size(corec, 3)

		coreS = reshape(coreS, [rk_1, 2, 2, 2, rk]...)
        
        core = reshape(permutedims(coreS, [1,3,4,5,2]), [rk*4*rk_1, 2]...) * reshape(permutedims(corec, [2, 1, 3]), [2, Rk_1 * Rk]...)
        
		core = reshape(core, [rk_1, 2, 2, rk, Rk_1, Rk]...)
        
		core = reshape(permutedims(core, [1, 3, 4, 5, 6, 2]), [rk_1*2*rk*Rk_1*Rk, 2]...) * reshape(permutedims(corec, [2, 1, 3]), [2, Rk_1 * Rk]...);
        
		core = reshape(permutedims(reshape(core, [rk_1, 2, rk, Rk_1, Rk, Rk_1, Rk]...), [1, 4, 6, 2, 3, 5, 7]), [rk_1*Rk_1*Rk_1, 2, rk*Rk*Rk]...);
        
		cells[k] = core
	end
    TTtensor(cells,false,false)
end

function calc_smol_grad(ctt, Stt, ytt)
	S_cores = Stt.cores
	c_cores = ctt.cores
	y_cores = ytt.cores
	dn 		= size(S_cores, 1)
	cells = CoreCell(undef, dn)

	for k = 1 : dn
		coreS = S_cores[k]
    	corec = c_cores[k]
		corey = y_cores[k]
		
		rk_1 = size(coreS, 1)
        rk   = size(coreS, 3)
        Rk_1 = size(corec, 1)
        Rk   = size(corec, 3)
		ryk  = size(corey, 3)
		ryk_1= size(corey, 1)

		coreS = reshape(coreS, [rk_1, 2, 2, 2, rk]...)
        
        core = reshape(permutedims(coreS, [1,3,4,5,2]), [rk*4*rk_1, 2]...) * reshape(permutedims(corec, [2, 1, 3]), [2, Rk_1 * Rk]...)
        
		core = reshape(core, [rk_1, 2, 2, rk, Rk_1, Rk]...)
        
		core = reshape(permutedims(core, [1, 2, 4, 5, 6, 3]), [rk_1*2*rk*Rk_1*Rk, 2]...) * reshape(permutedims(corey, [2, 1, 3]), [2, ryk_1 * ryk]...);
        
		core = reshape(permutedims(reshape(core, [rk_1, 2, rk, Rk_1, Rk, ryk_1, ryk]...), [1, 4, 6, 2, 3, 5, 7]), [rk_1*Rk_1*ryk_1, 2, rk*Rk*ryk]...);
        
		cells[k] = core
	end

	TTtensor(cells,false,false)
end