% script file for plotting final results
% make sure to run gridworldQLGPmain_test.m in each folder to get the
% results for different algorithms in the first place.
% author: Miao Liu, Girish Chowdhardy
% 
close all,clear,clc,

labels = {'k-','g-','c-','b-','r-','r-','y-','r*-','k*-'};

method = [0,1,3,6,7];%0-tabular, 1-function approximation with fixed bases, 3-GP-epsilon greedy, 
                      %6-GQ, 7-GP with variance backup
reward = [1];
scale = 1;
for rn = [1]
    f = figure(rn);
    for mn = 1 : length(method)
        if mn< length(method)+1
            if method(mn) == 0
                load([pwd,'/tabular/m0bw1am1nn1bg1to1_save']);
            elseif method(mn) == 1
                load([pwd,'/funcApprox/m1bw2am1nn1bg1to1_save']);
            elseif method(mn) == 3
                load([pwd,'/gp/m3bw2am1nn3bg1to2_save']);
            elseif method(mn) == 6
                load([pwd,'/funcApprox/m6bw3am1nn1bg1to1_save']);
            elseif method(mn) == 7
                load([pwd,'/gp/m7bw3am1nn4bg1to2_save']);
            end
        H(mn) = shadedErrorBar((1:scale:length(test_reward_mean))*50,test_reward_mean(1:scale:end), 0.4*test_reward_std(1:scale:end),labels{mn},1); hold on;
        hold on;
        else
            plot(t(1:scale:end),0.4783*ones(1,length(t(1:scale:end))))
        end
    end
    
    filename = ['gridworld_sdr'];
    xlim([0,1550])
    ylim([0,0.5])
    xlabel('Number of samples','fontsize',20)
    ylabel({'Sum of discounted reward'},'fontsize',20)
    
    leg = legend([H(1:5).mainLine],'QL-tab','QL-FB','GPQ-\epsilon-greedy', 'GQ','GPQ-optimistic');
    set(leg,'fontsize',15,'location','southeast','Box','off')
    set(gca,'fontsize',20);
    title('Gridworld')
    fileName = [pwd,'/',filename,'.pdf'];
    set(gcf, 'PaperPosition', [0 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
    set(gcf, 'PaperSize', [6 5]); %Set the paper to have width 6 and height 5.
    set(gcf, 'color', 'none')
    saveas(f, fileName);
end

for rn = [2]
    f = figure(rn);
    for mn = 1 : length(method)
        if mn< length(method)+1
            if method(mn) == 0
                load([pwd,'/tabular/m0bw1am1nn1bg1to1_save']);
            elseif method(mn) == 1
                load([pwd,'/funcApprox/m1bw2am1nn1bg1to1_save']);
            elseif method(mn) == 3
                load([pwd,'/gp/m3bw2am1nn3bg1to2_save']);
            elseif method(mn) == 6
                load([pwd,'/funcApprox/m6bw3am1nn1bg1to1_save']);
            elseif method(mn) == 7
                load([pwd,'/gp/m7bw3am1nn4bg1to2_save']);
            end
            H(mn) = shadedErrorBar((1:scale:length(test_step_mean))*50,test_step_mean(1:scale:end), 0.4*test_step_std(1:scale:end),labels{mn},1); hold on;
        else
            %plot(t(1:scale:end),0.4783*ones(1,length(t(1:scale:end))))
        end
    end
    filename = ['gridworld_stg'];
    xlim([0,1550])
    ylim([0,150])
    xlabel('Number of samples','fontsize',20)
    ylabel({'Steps to reach the goal'},'fontsize',20)
    leg = legend([H(1:5).mainLine],'QL-tab','QL-FB','GPQ-\epsilon-greedy', 'GQ','GPQ-optimistic');
    set(leg,'fontsize',15,'location','northeast','Box','off')
    set(gca,'fontsize',20);
    title('Gridworld')
    fileName = [pwd,'\',filename,'.pdf'];
    set(gcf, 'PaperPosition', [0 0 6 5]); %Position plot at left hand corner with width 5 and height 5.
    set(gcf, 'PaperSize', [6 5]); %Set the paper to have width 6 and height 5.
    set(gcf, 'color', 'none')
    saveas(f, fileName);
end