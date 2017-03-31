function [testReward_mean,testStep_mean,rew_eval] = Pendulum_test(params,N_eval,N_eps_length_test,gpr,theta)           

s_init = params.s_init;
statelist = params.statelist;
actionlist = params.actionlist;
method = params.approximation_on;
gamma = params.gamma;
%evaluate

rew_eval = zeros(1,N_eval);
eval_step = zeros(1,N_eval);

    for eval_count = 1:N_eval;
        
        x_test = params.s_init;
        s_test = DiscretizeState(x_test',statelist);
        for step_count = 1:N_eps_length_test

            if method == 0
            [Q_opt,action] = Q_greedy_act(theta,s_test,params,gpr);
            else
            [Q_opt,action] = Q_greedy_act(theta,x_test,params,gpr);
            end

            x_test = x_test';
            [xp_test, cost_test, endsim] = pendulum_simulator(x_test,action);
            x_test = xp_test';
            if abs(xp_test(1))> pi/2
                break;
            end
            if method==0
            s_test  = DiscretizeState(xp_test,statelist);
            end
            
            rew_eval(eval_count) = rew_eval(eval_count) + gamma^(step_count-1)*cost_test;            

        end
        eval_step(eval_count) = step_count;
    end
    testReward_mean = mean(rew_eval);
    testStep_mean = mean(eval_step);
    