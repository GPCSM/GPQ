clear,clc,close all
method_all = [1,6];
gpbandwidth_all = [0.5,1];
gpamptitude_all = [1];
gpnoise_all = [10^-3];
budget_all = [36,100];
tol_all = [0.1];
explore_dec = [0.5];
funcName = 'puddleWorldMain';
addpath('C:\Miao Liu\svn2\code\GPQlearning\');
addpath('C:\Miao Liu\svn2\code\GPQlearning\PuddleWorld');
breakflag = 0;
N_exec=20;
%statelist   = BuildStateList();  % the list of states
%[N_state,n_state_dim] = size(statelist); % Square state space
labels= {'r-','g-','b-'};
f = figure(1);
for nbw = 1 : length(gpbandwidth_all)
    for na = 1 : length(gpamptitude_all)
        for nn = 1 :length(gpnoise_all)
            for nbg = 1 : length(budget_all)
                N_state = budget_all(nbg);
                [X,Y] = meshgrid(1:sqrt(N_state),1:sqrt(N_state));
                %X = reshape(statelist(:,1),sqrt(N_state),sqrt(N_state));
                %Y = reshape(statelist(:,2),sqrt(N_state),sqrt(N_state));
                for nt = 1 : length(tol_all)
     
                        for nm = 1: length(method_all)
                                        clf;
                            for ni = 1 : length(explore_dec)
                                folderName = ['m',num2str(nm),'bw',num2str(nbw),'am',num2str(na)...
                                    'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'ex',num2str(ni)];
                                try
                                    load(folderName)
                                catch
                                    breakflag = 1;
                                end
                                for i = 1 : N_exec
                                    if i==1
                                        try
                                            V = value(i).V;
                                        catch
                                            iii=0;
                                        end
                                    else
                                        V = V + value(i).V;
                                    end
                                    pos = length(test(i).Reward_mean);
                                    len(i) = sum(pos);
                                end
                                len_min = min(len);
                                testRewardSum = zeros(N_exec,len_min);
                                testReward = zeros(N_exec,len_min);
                                testStep = zeros(N_exec,len_min);
                                trainReward_all = zeros(N_exec,201);
                                trainSteps_all = zeros(N_exec,201);
                                for i = 1 : N_exec
                                    pos = length(test(i).Reward_mean);
                                    val = test(i).Reward_mean(1:pos);
                                    testReward(i,1:len_min) = val(1:len_min);
                                    val = test(i).Steps_mean(1:pos);
                                    testStep(i,1:len_min) = val(1:len_min);
                                    %                                    testRewardSum(i,1:len_min) = test(i).sumReward_mean(1:len_min);
                                    %trainReward_all(i,:) = trainReward;%train(i).trainReward;
                                    %trainSteps_all(i,:) = trainSteps;%train(i).trainSteps;
                                    %tresRewardsum(i,:) = test(i).
                                end
                                Z = reshape(V/5,sqrt(N_state),sqrt(N_state));
                                train_reward_mean = mean(trainReward, 1);
                                train_reward_std = std(trainReward, 1);
                                train_step_mean = mean(trainSteps,1);
                                train_step_std = std(trainSteps,1);
                                test_reward_mean = mean(testReward, 1);
                                test_reward_std = std(testReward, 1);
                                test_step_mean = mean(testStep,1);
                                test_step_std = std(testStep,1);
                                %test_rewardsum_mean = mean(testRewardSum, 1);
                                %test_rewardsum_std = std(testRewardSum, 1);
                                scaling = 5;
                                len = length(train_reward_mean)-1;
                                if ni==1
                                    subplot(331),surf(X,Y,Z),hold on; title('value function (GPQ-1v)'),colorbar;
                                    zlim([-50,10]);
                                elseif ni ==2
                                    subplot(334),surf(X,Y,Z),hold on; title('value function (GPQ-2v)'),colorbar;
                                    zlim([-50,10]);
                                else
                                    subplot(337),surf(X,Y,Z),hold on; title('value function (GPQ-3v)'),colorbar;
                                    zlim([-50,10]);
                                end
                                subplot(332),errorbar(1:scaling:len,...
                                    train_reward_mean(1:scaling:len), 0.3*train_reward_std(1:scaling:len),labels{ni}),hold on;title('training reward');
                                xlim([0,len])
                                %ylim([-150,-30]);
                                subplot(333),errorbar(1:scaling:len,...
                                    train_step_mean(1:scaling:len), 0.3*train_step_std(1:scaling:len),labels{ni}),hold on;title('training steps');
                                xlim([0,len])
                                %ylim([50,500])
                                %subplot(334),plot(1:j,exp_rate(1:j)),title('exploration rate');
                                len = length(test_reward_mean);
                                scaling = 2;
                                subplot(335),errorbar(1:scaling:len,...
                                    test_reward_mean(1:scaling:len), test_reward_std(1:scaling:len),labels{ni}),hold on;title('test reward');
                                xlim([0,len])
                               % ylim([-160,20]);
                                subplot(336),errorbar(1:scaling:len,...
                                    test_step_mean(1:scaling:len), test_step_std(1:scaling:len),labels{ni}),hold on;title('test steps');
                                xlim([0,len])
                                %subplot(337),errorbar(1:scaling:len,...
                                % test_rewardsum_mean(1:scaling:len), test_rewardsum_std(1:scaling:len),labels{ni}),hold on;title('sum of test reward');
                                xlim([0,len])
                               ylim([0,1100])
                                savefilename = ['m',num2str(method_all(nm)),'bw',num2str(nbw),'am',num2str(na)...
                                    'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'_save'];
                                save(savefilename,'train_step_mean','test_step_mean','train_step_std','test_step_std',...
                                    'train_reward_mean','test_reward_mean','train_reward_std','test_reward_std','Z');
                                if breakflag==1
                                    breakflag = 0;
                                    break;
                                end
                                end
                                % subplot(337),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','South');set(h_legend,'FontSize',5);
                                %subplot(333),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','North');set(h_legend,'FontSize',5);
                                %subplot(335),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','South');set(h_legend,'FontSize',10);
                                %subplot(336),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','South');set(h_legend,'FontSize',10);
                                set(gcf, 'PaperPosition', [0 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
                                set(gcf, 'PaperSize', [8 5]); %Set the paper to have width 5 and height 5.
                                set(gcf, 'color', 'none')
                                saveFileName = ['m',num2str(nm),'bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'.pdf'];
                                saveas(f, saveFileName);
                    end
                end
            end
        end
    end
end
