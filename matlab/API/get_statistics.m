function [mean,var] = get_statistics(list)


mean_old = 0;
mean = 0;
std = 0;

for k = 1:length(list)
          
    mean = mean_old + (list(k) - mean_old)/k;
       
    
    std = ((k-1)*std + (list(k) - mean)*(list(k) - mean_old))/(k);
        
    mean_old = mean;
    
end

var = sqrt(std);

end