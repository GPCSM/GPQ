function phi_sa = Q_tabular_feature(s,a,params)


%N_grid_x = params.N_grid_x;
%N_grid_y = params.N_grid_y;
N_phi_s = params.N_phi_s;
N_phi_sa = params.N_phi_sa;


%{
    if s(1) > N_grid_x
        s(1) = N_grid_x;
    end
    if s(2) > N_grid_y
        s(2) = N_grid_y;
    end
    s_index = sub2ind([N_grid_x N_grid_y],...
    s(1),s(2));
%}

phi_s = zeros(N_phi_s,1);

    phi_s(s) = 1;

phi_sa = zeros(N_phi_sa,1);

    phi_sa(((a-1)*N_phi_s + 1):a*N_phi_s) = phi_s;     

end