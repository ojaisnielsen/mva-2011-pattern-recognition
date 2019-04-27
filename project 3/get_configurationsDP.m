function [locations_DP,scores_DP,part_locations_DP,messc] = get_configurationsDP(unary_dense,relations,nconfigurations,nmx_h,nmx_v,scs_wt);
%% Dynamic-programming based detection using  Felzenszwalb's code for Generalized Distance Transforms
nparts  = length(relations);
[sz_v,sz_h] =size(unary_dense{1});
nscales=  length(scs_wt);

ix = cell(nparts);
iy = cell(nparts);
merit = zeros(sz_v,sz_h,nscales);
mess = cell(nparts);
mess_p = cell(nparts);
beleif = cell(nparts);
part_names = {'left eye','right eye','nose','left mouth','right mouth'};

for sc_ind = 1:length(scs_wt),
    scl = scs_wt(sc_ind);
    %% turn the gaussian potential parameters into a cost
    sm = 0;
    %f = 1; figure
    for part = 1:nparts
        cens   = relations(part).center*scl;
        scales = relations(part).scale*scl;
        
        %% turn parameters of Gaussian into terms that facilitate
        %% the Distance Transform-based optimization.
        mh = cens(1);   mv = cens(2);
        sh = scales(1);  sv = scales(2);
        
        coeff_h_sq = 1/(2*sh^2);        coeff_v_sq = 1/(2*sv^2);
        coeff_h_un = -2*mh/(2*sh^2);    coeff_v_un = -2*mv/(2*sv^2);
        
        %% Generalized Distance Transforms
        [mess{part},ix{sc_ind,part},iy{sc_ind,part}]  = dt(1*double(unary_dense{part}),coeff_h_sq,coeff_h_un,coeff_v_sq,coeff_v_un);
        sm = sm + mess{part};
%         if (part ~= 3)
%             subplot(3, nparts, f), imagesc(exp(mess{part})), title(sprintf('DP, %s to nose, scale %d', part_names{part}, sc_ind));
%         end
%         f = f + 1;
    end
    merit(:,:,sc_ind) = sm;    

%     for part = 1:nparts
%         if (part ~= 3)
%             cens   = relations(part).center*scl;
%             scales = relations(part).scale*scl;
%             mh = cens(1);   mv = cens(2);
%             sh = scales(1);  sv = scales(2);
% 
%             coeff_h_sq = 1/(2*sh^2);        coeff_v_sq = 1/(2*sv^2);
%             coeff_h_un = -2*mh/(2*sh^2);    coeff_v_un = -2*mv/(2*sv^2);
% 
%             cost = zeros(sz_v,sz_h);
%             for part_p = 1:nparts,
%                 if (part_p ~= part)
%                     cost = cost + mess{part_p};
%                 end                
%             end
% 
%             [mess_p{part},~,~]  = dt(cost,coeff_h_sq,coeff_h_un,coeff_v_sq,coeff_v_un);
%             subplot(3, nparts, f), imagesc(exp(mess_p{part})), title(sprintf('DP, nose to %s, scale %d', part_names{part}, sc_ind));
%         end
%         f = f + 1;
%     end
%     
%     for part = 1:nparts
%         if (part ~= 3)
%             beleif{part} = 1*double(unary_dense{part}) + mess_p{part};
%         else
%             beleif{3} = merit(:,:,sc_ind);
%         end
%         subplot(3, nparts, f), imagesc(exp(beleif{part})), title(sprintf('DP, beleif at %s, scale %d', part_names{part}, sc_ind)); 
%         f = f + 1;
%     end
        
end



nmx_v = nmx_v + 1;
nmx_h = nmx_h + 1;
siz = size(merit);
for conf = 1:nconfigurations
    
    msm =  max(merit(:));
    idx = find(merit==msm); idx = idx(1);
    [p_v,p_h,sc] = ind2sub(siz,idx(1));
    sc = sc(1); p_v = p_v(1); p_h = p_h(1);
    
    for part =1:nparts,
        idxh(1,part) = ix{sc,part}(p_v,p_h);
        idxv(1,part) = iy{sc,part}(p_v,p_h);
    end
    
    scores_DP(conf)             = msm;
    locations_DP(:,conf)        = [p_h;p_v];
    part_locations_DP(:,:,conf) = [idxh;idxv];
    
    v_range = max(min(p_v+[-nmx_v:nmx_v],sz_v),1);
    h_range = max(min(p_h+[-nmx_h:nmx_h],sz_h),1);
    merit(v_range,h_range,:) = -10e5;
end

% function vec  = s_to_vec(m,s);
% vec = -[-2*m/s^2 ,1/s^2];
%
% function [m,s] = vec_to_s(vec);
% if size(vec,1)~=2, vec = vec'; end
% a  = -vec(2,:);  b  = -vec(1,:);
% s = sqrt(1./a);  m = -(b.* s.^2)/2;