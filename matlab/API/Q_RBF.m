function phi_sa = Q_RBF(s,a,params)

N_phi_s = params.N_phi_s;
N_phi_sa = params.N_phi_sa;

%% Enter RBF centers here (there must be N_phi_s centers)
centers = params.rbf_c;
% RBF variance
mu = params.rbf_mu;
if params.state_action_slicing_on==1 %% state-space slice approximation

    phi_s = zeros(N_phi_s,1);
    phi_s(1)=params.bw;
    for i = 1:N_phi_s-1
    %ss = params.statelist(s,:)';
    phi_s(1+i) = exp(-0.5*norm(s - centers(:,i))^2/mu(i));
    end
%Output RBF phi_sa
    phi_sa = zeros(N_phi_sa,1);
    phi_sa(((a-1)*N_phi_s + 1):a*N_phi_s) = phi_s;

elseif state_action_slicing_on==0%% centers spread over X \times A

   
    phi_sa = zeros(N_phi_s,1);
    phi_sa(1)=params.bw;
    x=[s;a];
    for i = 1:N_phi_s-1
        phi_sa(1+i) = exp(-0.5*norm(x - centers(:,i))^2/mu(i));
    end
end

end