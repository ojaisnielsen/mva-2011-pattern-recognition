function [locations_DP,scores_DP,part_locations_DP,messc] = get_configurationsFL(unary_dense,relations,nconfigurations,nmx_h,nmx_v,scs_wt);

nparts  = length(relations);
[sz_v,sz_h] =size(unary_dense{1});
nscales=  length(scs_wt);


[P_v, ~, R_v] = ndgrid(1:sz_v, 1:sz_h, 1:sz_v);
a = sz_v - P_v + 1 - R_v;
[~, P_h, R_h] = ndgrid(1:sz_v, 1:sz_h, 1:sz_h);
b = sz_h - P_h + 1 - R_h;

ix = cell(nparts);
iy = cell(nparts);
merit = zeros(sz_v,sz_h, nscales);
mess = cell(nparts);
mess_p = cell(nparts);
beleif = cell(nparts);
part_names = {'left eye','right eye','nose','left mouth','right mouth'};

for sc_ind = 1:length(scs_wt),
    f = 1; figure
    for part = 1:nparts,
        ix{part} = zeros(sz_v,sz_h,sc_ind);
        iy{part} = zeros(sz_v,sz_h,sc_ind);

        scl = scs_wt(sc_ind);
        cens   = relations(part).center*scl;
        scales = relations(part).scale*scl; 
        mh = cens(1);   mv = cens(2);
        sh = scales(1);  sv = scales(2); 
        
        c = (a - mv).^2 / (2 * sv^2);
        d = (b - mh).^2 / (2 * sh^2);
        
        for r_h = 1:sz_h,
            100*r_h/sz_h
            for r_v = 1:sz_v,
                pot = unary_dense{part} - c(:,:,r_v) - d(:,:,r_h);
                [m, I] = max(pot(:));
                mess{part}(sz_v - r_v + 1, sz_h - r_h + 1) = m;
                [iy{part}(r_v, r_h, sc_ind), ix{part}(r_v, r_h, sc_ind)] = ind2sub([sz_v,sz_h], I);
            end
        end
        if (part ~= 3)
            subplot(3, nparts, f), imagesc(exp(mess{part})), title(sprintf('FL, %s to nose, scale %d', part_names{part}, sc_ind));
        end
        f = f + 1;        
        merit(:,:,sc_ind) = merit(:,:,sc_ind) + mess{part};
    end

    
    for part = 1:nparts
        if (part ~= 3)
            cens   = relations(part).center*scl;
            scales = relations(part).scale*scl;
            mh = cens(1);   mv = cens(2);
            sh = scales(1);  sv = scales(2);

            coeff_h_sq = 1/(2*sh^2);        coeff_v_sq = 1/(2*sv^2);
            coeff_h_un = -2*mh/(2*sh^2);    coeff_v_un = -2*mv/(2*sv^2);
            c = (a - mv).^2 / (2 * sv^2);
            d = (b - mh).^2 / (2 * sh^2);

            cost = zeros(sz_v,sz_h);
            for part_p = 1:nparts,
                if (part_p ~= part)
                    cost = cost + mess{part_p};
                end                
            end

            for p_h = 1:sz_h,
                100*r_h/sz_h
                for p_v = 1:sz_v,
                    pot = cost - c(:,:,p_v) - d(:,:,p_h);
                    mess_p{part}(sz_v - p_v + 1, sz_h - p_h + 1) = max(pot(:));                    
                end
            end
            subplot(3, nparts, f), imagesc(exp(mess_p{part})), title(sprintf('FL, nose to %s, scale %d', part_names{part}, sc_ind));
        end
        f = f + 1;
    end
    
    for part = 1:nparts
        if (part ~= 3)
            beleif{part} = 1*double(unary_dense{part}) + mess_p{part};
        else
            beleif{3} = merit(:,:,sc_ind);
        end
        subplot(3, nparts, f), imagesc(exp(beleif{part})), title(sprintf('FL, beleif at %s, scale %d', part_names{part}, sc_ind)); 
        f = f + 1;
    end    
    
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
        idxh(1,part) = ix{part}(p_v,p_h,sc);
        idxv(1,part) = iy{part}(p_v,p_h,sc);
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
