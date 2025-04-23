module GSSPAlgorithm
    export GSSP
    ###############################################################################
    # 1)  P_{L_k}(w): keep only the top-k entries of w, zero out the rest
    ###############################################################################
    function PLk(w::Vector{Float64}, k::Int)
        """
        P_{L_k}(w) as in Kyrillidis et al. (2013):
        Keeps the k largest entries of w (descending order),
        and sets all other entries to zero.
        """
        n = length(w)
        if k >= n
            return copy(w)  # no need to truncate
        end
        # Find indices of the top k entries (descending)
        topk_idx = partialsortperm(w, 1:k; rev=true)
        w_copy   = zeros(n)
        w_copy[topk_idx] = w[topk_idx]
        return w_copy
    end

    ###############################################################################
    # 2)  P_{λ}^+(w): Euclidian projector onto { x ≥ 0, sum(x)=λ }
    #     Definition 2.2 in the paper
    ###############################################################################
    function P_lambda_plus(w::Vector{Float64}, λ::Float64; verbose::Bool=true)
        """
        P_{λ}^+(w) from Definition 2.2:
        (P_{λ}^+(w))_i = max(w_i - τ, 0),
        where τ = (1 / ρ) * (sum_{i=1}^ρ w_i - λ)
        and
        ρ = max{ j : w_j > (1/j)* ( (sum_{i=1}^j w_i) - λ ) }.
        """
        # Cap w to avoid numerical instability
        w_max = 1e6  # Choose an appropriate cap based on expected scale
        w = min.(w, w_max)

        
        # Sort w in descending order
        w_sorted = sort(w, rev=true)
        partial_sum = 0.0
        ρ = 1

        # ρ is the largest index such that w_sorted[ρ] > ( (Σ_{j≤ρ} w_sorted[j] - λ ) / ρ )
        for j in eachindex(w_sorted)
            partial_sum += w_sorted[j]
            t_j = (partial_sum - λ)/j
            if w_sorted[j] > t_j
                ρ = j
            end
        end

        # Shift τ = ( sum_{j=1 to ρ} w_sorted[j] - λ ) / ρ
        if ρ == 0
            # Edge case: if even the largest w_sorted is <= (partial_sum - λ)/1,
            # then the entire vector is truncated to 0, or λ=0
            return zeros(length(w))
        end
        τ = (sum(w_sorted[1:ρ]) - λ)/ρ

        # Final: clamp each coordinate: [w_i - τ]_+
        # (still unsorted, so we apply it to w in original order!)
        #print difference between w and τ
        if verbose
            println("w - τ: ", w .- τ)
        end
        return max.(w .- τ, 0)
    end

    ###############################################################################
    # 3)  Algorithm 1 (GSSP):  β = GSSP(w, k, λ)
    ###############################################################################
    function GSSP(w::Vector{Float64}, k::Int, λ::Float64; verbose::Bool=true)
        """
        Algorithm 1 (GSSP) from Kyrillidis et al. (2013):
        1) x = P_{L_k}(w)
        2) S = supp(x)
        3) β[S] = P_{λ}^+( w[S] ),   β[S^c] = 0
        """
        if verbose
            println("w: ", w)
        end
        # (1) Truncate w to top k entries
        x = PLk(w, k)

        if verbose
            println("x: ", x)
        end

        # (2) S = set of indices where x != 0
        S = findall(!iszero, x)

        if verbose
            println("S: ", S)
        end

        # (3) Project the subvector w[S] onto simplex { sum=λ, >=0 }
        w_sub  = w[S]               # original w restricted to S
        β_sub  = P_lambda_plus(w_sub, λ; verbose=verbose)
        if verbose
            println("w_sub: ", w_sub)
            println("β_sub: ", β_sub)
        end


        # Reassemble full β
        β = zeros(length(w))
        for (idx, val) in zip(S, β_sub)
            β[idx] = val
        end
        return β
    end

end