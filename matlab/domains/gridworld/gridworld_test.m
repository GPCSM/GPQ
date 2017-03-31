function [testReward_mean,testStep_mean,rew_eval] = gridworld_test(params,N_eval,N_eps_length_test,gpr,theta)

x_init = params.x_init;
gamma = params.gamma;
method = params.approximation_on;
statelist = params.statelist;
%evaluate
rew_eval = zeros(1,N_eval);
rew_eval_discount = zeros(1,N_eval);
eval_step = zeros(1,N_eval);
for eval_count = 1:N_eval;
    s_prev = x_init;
    s_prev_indx = 1;%DiscretizeState(s_prev',statelist);
    for step_count = 1:N_eps_length_test
        if method == 0
            [Q_opt,action] =  Q_greedy_act_test(theta,s_prev_indx,params,gpr);
        else
            [Q_opt,action] =  Q_greedy_act_test(theta,s_prev,params,gpr);
        end
        s_next = gridworld_trans(s_prev,action,params);
                        
        [rew,breaker] = gridworld_rew(s_next,params);
        %Calculate The Reward
        rew_eval_discount(eval_count) = rew_eval_discount(eval_count) + gamma^(step_count-1)*rew;
        rew_eval(eval_count) = rew_eval(eval_count) + rew;
        if breaker
           break;
        end
        s_prev = s_next;
        s_indx_new = DiscretizeState(s_prev',statelist);
        if method == 0
            s_prev_indx = s_indx_new;
        end
    end    
    eval_step(eval_count) = step_count;
end

testReward_mean = mean(rew_eval_discount);
testStep_mean = mean(eval_step);
