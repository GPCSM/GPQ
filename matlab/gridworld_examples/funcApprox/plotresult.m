clear,clc,close all
method_all = [1,6];
gpbandwidth_all = [1];
gpamptitude_all = [1];
gpnoise_all = [1];
N_grid_all = [5];%domain size
tol_all = [1];
%funcName = 'puddleWorldMain';
%addpath('C:\Miao Liu\svn2\code\GPQlearning\');
%addpath('C:\Miao Liu\svn2\code\GPQlearning\PuddleWorld');
breakflag = 0;
N_exec=2;
%statelist   = BuildStateList();  % the list of states
%[N_state,n_state_dim] = size(statelist); % Square state space
labels= {'r-o','g-.','b*-'};
f = figure(1);
%for nbw = 1 : length(gpbandwidth_all)
for na = 1 : length(gpamptitude_all)
    for nn = 1 :length(gpnoise_all)
        for nbg = 1 : length(N_grid_all)
            [X,Y] = meshgrid(1:N_grid_all(nbg),1:N_grid_all(nbg));
            N_state = N_grid_all(nbg)*N_grid_all(nbg);
            %X = reshape(statelist(:,1),sqrt(N_state),sqrt(N_state));
            %Y = reshape(statelist(:,2),sqrt(N_state),sqrt(N_state));
            for nt = 1 : length(tol_all)
                clf;
                for nm = 1: length(method_all)
                    if nm == 1
                        nbw = 2;
                    else
                        nbw = 3;
                    end
                    folderName = ['m',num2str(method_all(nm)),'bw',num2str(nbw),'am',num2str(na)...
                        'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt)];
                    try
                        load(folderName)
                    catch
                        breakflag = 1;
                    end
                    for i = 1 : N_exec
                        if i==1
                            V = value(i).V;
                        else
                            V = V + value(i).V;
                        end
                        pos = length(test(i).sumReward_mean);
                        len(i) = sum(pos);
                    end
                    len_min = min(len);
                    testRewardSum = zeros(N_exec,len_min);
                    testReward = zeros(N_exec,len_min);
                    testStep = zeros(N_exec,len_min);
                    trainReward = zeros(N_exec,201);
                    trainSteps = zeros(N_exec,201);
                    for i = 1 : N_exec
                        pos = length(test(i).Reward_mean);
                        val = test(i).Reward_mean(1:pos);
                        testReward(i,1:len_min) = val(1:len_min);
                        val = test(i).Steps_mean(1:pos);
                        testStep(i,1:len_min) = val(1:len_min);
                        testRewardSum(i,1:len_min) = test(i).sumReward_mean(1:len_min);
                        trainReward(i,:) = train(i).trainReward;
                        trainSteps(i,:) = train(i).trainSteps;
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
                    test_rewardsum_mean = mean(testRewardSum, 1);
                    test_rewardsum_std = std(testRewardSum, 1);
                    scaling = 10;
                    len = length(train_reward_mean)-1;
                    if nm==1
                        subplot(331),surf(X,Y,Z),hold on; title('value function (GPQ-epsilon)'),colorbar;
                    else
                        subplot(334),surf(X,Y,Z),hold on; title('value function (GPQ-2var)'),colorbar;
                    end
                    subplot(332),errorbar(1:scaling:len,...
                        train_reward_mean(1:scaling:len), train_reward_std(1:scaling:len),labels{nm}),hold on;title('training reward');
                    xlim([0,len])
                    % ylim([-160,40]);
                    subplot(333),errorbar(1:scaling:len,...
                        train_step_mean(1:scaling:len), train_step_std(1:scaling:len),labels{nm}),hold on;title('training steps');
                    xlim([0,len])
                    %subplot(234),plot(1:j,exp_rate(1:j)),title('exploration rate');
                    len = length(test_reward_mean);
                    scaling = 2;
                    subplot(335),errorbar(1:scaling:len,...
                        test_reward_mean(1:scaling:len), test_reward_std(1:scaling:len),labels{nm}),hold on;title('test reward');
                    xlim([0,len])
                    subplot(336),errorbar(1:scaling:len,...
                        test_step_mean(1:scaling:len), test_step_std(1:scaling:len),labels{nm}),hold on;title('test steps');
                    xlim([0,len])
                    subplot(337),errorbar(1:scaling:len,...
                        test_rewardsum_mean(1:scaling:len), test_rewardsum_std(1:scaling:len),labels{nm}),hold on;title('sum of test reward');
                    xlim([0,len])
                    savefilename = ['m',num2str(method_all(nm)),'bw',num2str(nbw),'am',num2str(na)...
                        'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'_save'];
                    save(savefilename,'train_step_mean','test_step_mean','train_step_std','test_step_std',...
                        'test_rewardsum_mean','test_rewardsum_std',...
                        'train_reward_mean','test_reward_mean','train_reward_std','test_reward_std','Z');
                end
                if breakflag==1
                    breakflag = 0;
                    break;
                end
                subplot(337),h_legend=legend('Tab','Location','South');set(h_legend,'FontSize',5);
                %subplot(233),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','North');set(h_legend,'FontSize',5);
                %subplot(235),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','South');set(h_legend,'FontSize',10);
                %subplot(236),h_legend=legend('GPQ-epsilon','GPQ-2var','Location','South');set(h_legend,'FontSize',10);
                set(gcf, 'PaperPosition', [0 0 8 5]); %Position plot at left hand corner with width 5 and height 5.
                set(gcf, 'PaperSize', [8 5]); %Set the paper to have width 5 and height 5.
                set(gcf, 'color', 'none')
                saveFileName = ['bw',num2str(nbw),'am',num2str(na),'nn',num2str(nn),'bg',num2str(nbg),'to',num2str(nt),'.pdf'];
                saveas(f, saveFileName);
            end
        end
    end
end
%end
