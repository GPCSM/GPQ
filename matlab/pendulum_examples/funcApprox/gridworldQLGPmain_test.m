% script file for testing different algorithms under various parameter
% settings
% authors: Miao Liu, Girish Chowdhardy
%
clear, clc,
curDir = pwd;
mainDir = fileparts(fileparts(curDir));            % go up 2 directories
addpath(fullfile(mainDir,'API'));
addpath(fullfile(mainDir,'domains/Pendulum'));

rand('state',1000*sum(clock));
method_all = [1,6];  % 1- linear function approximation, 6-GQ
gpbandwidth_all = [0.5,1];
gpamptitude_all = [1];
gpnoise_all = [10^-3];
budget_all = [36,100];
tol_all = [0.1];
explore_dec = [0.5];
%---------------choose a specific setting to run---------------------------
nm = 1;
nbw = 1;
na = 1;
nn = 1;
nbg = 2;
nt = 1;
ne = 1;
method = method_all(nm);
%% Pendulum Parameters
dt = 0.1;           % Simulation step
noise = 10;         % Range of noise
allu = [-50 0 +50]; % The 3 actions in Newtons
x=[0;0];
x_init = x;
s_init = x;
u=0;
u_min=-50;
u_max=50;
u_step=50;
x_min=[-pi,-pi];
x_max=[pi,pi];

params.A = gpamptitude_all(na);
sigma = gpnoise_all(nn); % variance of GP
params.sigma = sigma;
params.rbf_mu= ones(1,2)*gpbandwidth_all(nbw); %bandwidth of GP
tol=tol_all(nt);
params.tol = tol;
N_budget = budget_all(nbg); % budget for GP
params.N_budget=N_budget;
params.N_state = N_budget;
params.var = 0;
dim_u=1;
u_grid=u_min:u_step:u_max;
u_grid_size=max(size(u_grid));
params.a_init = u;
params.u_grid = u_grid;
%% discretized parameters
statelist = BuildStateList_pendulum(sqrt(N_budget));  % the list of states
actionlist  = BuildActionList_pendulum(allu); % the list of actions
params.statelist = statelist;
params.actionlist = actionlist;
[N_state,n_state_dim] = size(statelist); % Square state space
params.N_state = N_state;
N_act = (u_max - u_min)/u_step + 1;

n_state_dim=2;

[nouse, a_init]= histc(0,u_grid); %just need the index


params.N_state_dim=n_state_dim;
params.N_act = N_act;
params.noise = noise;
params.x_min = x_min;
params.x_max = x_max;
params.u_min = u_min;
params.u_max = u_max;
params.x_init = x_init;
params.s_init = x_init;
params.u_init = a_init;
params.n_state_dim = n_state_dim;

%% Q Learning Parameters
gamma = 0.95; % Discount Factor
params.gamma=gamma;

N_eps_length_train = 50; % Length of an episode
N_eps_length_test = 1000;

N_eps_train = 600;
N_eps_test = 5; %200 Number of episodes

eval_freq = 200; % How frequently current policy should be evaluated (per step)

N_exec = 3; % Number of executions of the algorithm
N_eval = 10; % How many times should it be evaluated ?


alpha_init = 0.5; % Initial Learning Rate
alpha_dec = tol_all(nt); % Learning Rate Decay Rate
mu=1;
p_eps_init = 0.8; % Initial Exploration Rate
p_eps_dec = explore_dec(ne); % Exploration Rate Decay
var_explore_on=0;%explore based on the variance
params.mu_var_improv=0; %parameter for variance based exploration
params.alpha_init = alpha_init;
params.alpha_dec = alpha_dec;
%%%%% ## which algorithm to use ## %%%%%%%
approximation_on=method;%0 for tabular, 1 for RBF with copy-paste features, 2 for RBF with centers over actions,
%3 GP, 4 BKR-CL, 5 GP with hyperparameter learning
max_points=N_budget;%50 %max centres allowed for RBF Kernels

if method == 2
    cl_on = 1;
else
    cl_on=0;%is concurrent learning on?
end
params.cl_rate = .1; % Concurrent learning rate, set 0 for classical Q Learning
params.cl_rate_decrease=1.1;

params.approximation_on=approximation_on;
params.cl_on=cl_on;


%% diagnostic parameters
convergence_diagnostic_on=1;
%E_pi_phi=zeros(params.N_phi_sa,params.N_phi_sa);
%E_pi_phi_m=zeros(params.N_phi_sa,params.N_phi_sa);
testSteps_mean = [];
testSteps_std = [];
trainEpsNum = [];
%% Algorithm Execution
eval_counter = zeros(N_exec,1);
eval_num = 0;
% Execution Loop
rew_exec = [];

grid = statelist;
trainReward = zeros(N_exec,N_eps_train+1);
trainSteps =  zeros(N_exec,N_eps_train+1);
for i = 1 : N_exec
    train_Reward = zeros(1,N_eps_train+1);
    train_Steps = zeros(1,N_eps_train+1);
    sprintf('At execution %d \n',i)
    % Reset The Q function
    eval_num = 0;
    [params,gpr,theta] = QL_reset(params);
    step_counter = 0;
    rew_exec_ind = [];
    %eval_counter = 0;
    %--------------------------------online training---------------------------
    for j = 1:N_eps_train
        %fprintf('At episode %d/%d  of execution %d/%d
        %\n',j,N_eps_train,i,N_exec);
        % Reset the initial state
        s_old = s_init;
        x = x_init;
        %[x_disc, x_indx] = get_discrete_x(x,x_grid,y_grid,params);
        s_indx_old = DiscretizeState(x',statelist);
        for k = 1: N_eps_length_train
            % Is it evlauation time ?
            if(mod(step_counter,eval_freq) == 0)
                eval_num = eval_num + 1;
                [test_Reward_mean(eval_num),test_Steps_mean(eval_num),rew_eval] = Pendulum_test(params,N_eval,N_eps_length_test,gpr,theta);
                [rew_exec_ind(1,eval_num),nothing] = get_statistics(rew_eval);
            end%end eval loop
            %DCmotor_test;
            % Increment the step counter
            
            step_counter = step_counter + 1;
            
            % Set the Exploration rate
            p_eps = p_eps_init/...
                (step_counter)^p_eps_dec;
            
            if method~=7
                % Check if going to explore or not
                r = sample_discrete([p_eps 1-p_eps]);
                if r==1 % Explore
                    p = 1/N_act.*ones(1,N_act);
                    action_indx = sample_discrete(p);
                else % Exploit
                    if var_explore_on==1 && approximation_on==3 %take into consideration variance while selecting greedy action
                        [Q_opt,action_indx]=var_greedy_act(theta,s_old,params,gpr);
                    else %purely greedy
                        if method == 0
                            [Q_opt,action_indx] = Q_greedy_act(theta,s_indx_old,params,gpr);
                        else
                            [Q_opt,action_indx] = Q_greedy_act(theta,x,params,gpr);
                        end
                    end
                end
            else
                [Q_opt,action_indx] = Q_greedy_act(theta,x,params,gpr);
            end
            % get samples from MDP
            x = x';
            [xp, rew, endsim] = pendulum_simulator(x,action_indx);
            train_Reward(1,j) = train_Reward(1,j) + rew;
            train_Steps(1,j) = k;
            
            x = xp';
            if abs(x(1))> pi/2
                break;
            end
            
            sp  = DiscretizeState(xp,statelist);
            %[x_disc, x_indx]=get_discrete_x(x,x_grid,y_grid, params);
            %x_disc = x';
            
            if method == 1||method == 6||method == 3||method == 7
                s_new = xp';
            else
                s_new = sp';
            end
            if method == 4
                [theta,params.o,params] = QL_update(step_counter,k,approximation_on,theta,s_new,s_old,s_indx_old,rew,action_indx,params,params.o);
            else
                [theta,gpr,params] = QL_update(step_counter,k,approximation_on,theta,s_new,s_old,s_indx_old,rew,action_indx,params,gpr);
            end
            s_old = s_new;
            s_indx_old = sp;
        end
        if method == 3 || method == 7
            alpha = gpr.get('alpha_store');
            if j>1
                diff_alpha_norm(j-1) = norm(alpha-alpha_old,'fro');
            end
            alpha_old = alpha;
        end
        disp(['training episodes:', num2str(j), '/', num2str(N_eps_train)])
    end
    %plot value function
    if method == 3 || method == 7
        state = gpr.get('BV_store');
        gp = gpr.get('obs_store');
        store_points = gpr.get('current_size');
        
        mean_post = zeros(3,size(N_state,2));
        var_post = zeros(3,size(N_state,2));
        for a = 1 : params.N_act
            %--------gp
            states=[];
            index=1;
            for ii=1:params.N_state
                %for jj=1:params.N_state
                x_input = [params.statelist(index,:)';a];
                [mean_post(a,index), var_post(a,index)] = gpr.predict(x_input,params);
                index=index+1;
                % end
            end
        end
        V = max(mean_post,[],1);
        %V = min(V,0);
    elseif method == 0
        Q = reshape(theta,params.N_state,params.N_act);
        V = max(Q,[],2);
    elseif method == 1||method == 6
        for a = 1 : 3
            %--------gp
            states=[];
            index=1;
            for ii=1:params.N_state
                x_input = [params.statelist(index,:)'];
                phi_sa = Q_RBF(x_input,a,params);
                index=index+1;
                Q(ii,a) = theta'*phi_sa;
            end
        end
        V = max(Q,[],2);
    end
    
    
    %save results
    trainReward(i,:) = train_Reward;
    trainSteps(i,:) = train_Steps;
    if method==3||method==7
        test(i).diff_alpha = diff_alpha_norm;
    end
    test(i).Steps_mean = test_Steps_mean;
    test(i).Reward_mean = test_Reward_mean;
    %test(i).sumReward_mean = rew_exec_ind;
    value(i).V = V;
end
filename = ['m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'ex',num2str(ne)];
save(filename,'test','trainReward','trainSteps','value');

