% script file for testing different algorithms under various parameter
% settings
% author: Miao Liu, Girish Chowdhardy
%

clear, clc,
curDir = pwd;
mainDir = fileparts(fileparts(curDir));            % go up 2 directories
addpath(fullfile(mainDir,'API'));
addpath(fullfile(mainDir,'domains/gridworld'));

rand('state',1000*sum(clock));

method_all = [3,7]; %3-epsilon greedy, 7-optimistism under uncertainty
gpbandwidth_all = [0.5,1,2];
gpamptitude_all = [1];
gpnoise_all = [10^-3,10^-2,10^-1,0^0];
N_grid_all = [5,10];%domain size
tol_all = [10^-2,10^-4];
nm = 1;
if nm == 1
    nbw = 2;
    nn = 3;
elseif nm ==2
    nbw =3;
    nn = 4;
end
na = 1;
nbg = 1;
nt = 2;
%% Gridworld Parameters

N_grid = N_grid_all(nbg); % Number of grids on one direction
N_state = N_grid*N_grid; % Square Gridworld
N_act = 5; % 4 directions + null action

n_state_dim=2;
params.n_state_dim = n_state_dim;
params.N_state = N_state;
s_init = [1;1]; % Start in the top left corner
a_init=2;
s_goal = [N_grid;N_grid]; % Goal is in the lower right corner
params.s_init = s_init;
params.x_init = s_init;
params.a_init = a_init;
params.u_init = a_init;
N_obstacle = 0; % Number of obstacles

obs_list = [];%[3,4;2,2]; % Coordinates of obstacles

rew_goal = 1; % Reward for getting to the goal
rew_obs = -10; % Cost for crashing to an obstacle

noise = 0.1; % Probability of stepping in the wrong direction

params.A = gpamptitude_all(na);
sigma = gpnoise_all(nn); % variance of GP
params.sigma = sigma;
params.rbf_mu= ones(1,2)*gpbandwidth_all(nbw); %bandwidth of GP
tol=tol_all(nt);
params.tol = tol;
[Y,X] = meshgrid(1:N_grid,1:N_grid);
statelist = [X(:),Y(:)];
params.statelist = statelist;
params.N_grid = N_grid;
params.s_goal = s_goal;
params.rew_goal = rew_goal;
params.rew_obs = rew_obs;
params.N_state_dim=n_state_dim;
params.N_act = N_act;
params.noise = noise;
params.obs_list = obs_list;
params.N_obstacle = N_obstacle;
params.tol=tol_all(nt);%1e-4;
%% Q Learning Parameters
gamma = 0.9; % Discount Factor
params.gamma=gamma;

N_eps_length_train = 200; % Length of an episode
N_eps_length_test = 200;
N_eps_train = 200;
N_eval = 10; %200 Number of episodes

eval_freq = 100; % How frequently current policy should be evaluated (per step)
N_eval = 10; % How many times should it be evaluated ?


N_budget =  N_state; % Max Number of points to be kept in history stack

params.N_budget=N_budget;
data_method = 2; % 1 For cylic and 2 for SVD
params.epsilon_data_select=0.2;
stack_index=0;
points_in_stack=0;

alpha_init = 0.5; % Initial Learning Rate
alpha_dec = 0.5; % Learning Rate Decay Rate
params.alpha_init = alpha_init;
params.alpha_dec = alpha_dec;

mu=1;
p_eps_init = 0.8; % Initial Exploration Rate
p_eps_dec = 0.1; % Exploration Rate Decay
var_explore_on=0;%explore based on the variance
params.mu_var_improv=0; %parameter for variance based exploration

%%%%% ## which algorithm to use ## %%%%%%%
approximation_on=method_all(nm);%0 for tabular, 1 for RBF with copy-paste features, 2 for RBF with centers over actions,
                         %3 GP, 4 BKR-CL, 5 GP with hyperparameter learning
method = method_all(nm);
max_points=N_budget;%50 %max centres allowed for RBF Kernels
if method == 2
    cl_on = 1;
else
    cl_on=0;%is concurrent learning on?
end
params.cl_rate = .1; % Concurrent learning rate, set 0 for classical Q Learning
cl_rate_decrease=1.1;

params.approximation_on=approximation_on;
params.cl_on=cl_on;

%savefile = ['tmp','m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'.mat'];
%save(savefile);
%trainReward = zeros(N_exec,N_eps_train+1);
%trainSteps =  zeros(N_exec,N_eps_train+1);
%individualRun(nm,nbw,na,nn,nbg,nt,1)



N_exec = 3;% total number of MC executions
train_Reward = zeros(1,N_eps_train+1);
train_Steps = zeros(1,N_eps_train+1);
for i = 1 : N_exec
    sprintf('At execution %d \n',i)
    % Reset The Q function
    eval_num = 0;
    [params,gpr,theta] = QL_reset(params);
    step_counter = 0;
    rew_exec_ind = [];
    %eval_counter = 0;
    %--------------------------------online training---------------------------
    for j = 1:N_eps_train
        % Reset the initial state
        s_old = s_init;
        s_indx_old = DiscretizeState(s_old',statelist);
        %disp(['episode:',num2str(j),'exploration rate:',num2str(p_eps)]);
        for k = 1: N_eps_length_train
            % Is it evlauation time ?
            if(mod(step_counter,eval_freq) == 0)
                eval_num = eval_num + 1;
                [test_Reward_mean(eval_num),test_Steps_mean(eval_num),rew_eval] = gridworld_test(params,N_eval,N_eps_length_test,gpr,theta);
                %[eval_counter,eval_num,rew_exec] =
                %Pendulum_test(step_counter,eval_counter,i,N_eval,params,N_eps_length_test,eval_num,rew_exec,gpr,theta);
                [rew_exec_ind(1,eval_num),nothing] = get_statistics(rew_eval);
            end%end eval loop
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
                            [Q_opt,action_indx] = Q_greedy_act(theta,s_old,params,gpr);
                        end
                    end
                end
            else
                [Q_opt,action_indx] = Q_greedy_act(theta,s_old,params,gpr);
            end
            action = action_indx;
            % Next State
            
            s_new = gridworld_trans(s_old,action,params);
            s_indx_new = DiscretizeState(s_new',statelist);
            %Calculate The Reward
            [rew,breaker] = gridworld_rew(s_new,params);
            
            train_Reward(1,j) = train_Reward(1,j) + params.gamma^(k-1)*rew;
            train_Steps(1,j) = k;
            
            if method == 4
                [theta,params.o,params] = QL_update(step_counter,k,approximation_on,theta,s_new,s_old,s_indx_old,rew,action_indx,params,params.o);
            else
                if method == 0
                    [theta,gpr,params,y] = QL_update(step_counter,k,approximation_on,theta,s_indx_new,s_old,s_indx_old,rew,action_indx,params,gpr,breaker);
                else
                    [theta,gpr,params,y] = QL_update(step_counter,k,approximation_on,theta,s_new,s_old,s_indx_old,rew,action_indx,params,gpr,breaker);
                end
            end
            % if reachs the goal breaks the episode
            if breaker
                break;
            end
            
            s_old = s_new;
            if method == 0
                s_indx_old = s_indx_new;
            end
            
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
    train(i).trainReward = train_Reward;
    train(i).trainSteps = train_Steps;
    if method==3||method==7
        test(i).diff_alpha = diff_alpha_norm;
    end
    test(i).Steps_mean = test_Steps_mean;
    test(i).Reward_mean = test_Reward_mean;
    test(i).sumReward_mean = rew_exec_ind;
    value(i).V = V;
end
filename = ['m',num2str(method_all(nm)),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt)];
save(filename,'test','train','value');