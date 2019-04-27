  function features_aggregate = get_features(im)
        
        [features{1}] = detBG(im);
        [features{2}] = detTG(im);
        [features{3}] = detCG(im);             

ncues = 3;
total_features =0;
[size_m,size_n,nd] = size(im);
norient =  8;
features_aggregate(1:size_m,1:size_n,1:norient,1)  = 1;
         
for cue = 1:ncues
    [sz_hor,sz_ver,norient,nfeat] = size(features{cue});
    for ft = 1:nfeat
        total_features = total_features + 1;
        features_aggregate(:,:,:,total_features)  = features{cue}(:,:,:,ft);
    end
end



function [bg,theta] = detBG(im)
% function [bg,theta] = detBG(im,radius,norient)
% Compute smoothed but not thinned BG fields.
radius=0.01; norient=8; 

[h,w,unused] = size(im);
idiag = norm([h w]);
if isrgb(im), im=rgb2gray(im); end

% compute brightness gradient
[bg,theta] = cgmo(im,idiag*radius,norient,...
                  'smooth','savgol','sigmaSmo',idiag*radius);
              
function [tg,theta] = detTG(im)
% function [tg,theta] = detTG(im,radius,norient)
% Compute smoothed but not thinned TG fields.

radius=0.02;  norient=8; 
[h,w,unused] = size(im);
idiag = norm([h w]);
if isrgb(im), im=rgb2gray(im); end
% compute texture gradient
no = 6;ss = 1;ns = 2;sc = sqrt(2);el = 2; k = 64;
fname = sprintf( ...
    'unitex_%.2g_%.2g_%.2g_%.2g_%.2g_%d.mat',no,ss,ns,sc,el,k);
textonData = load(fname); % defines fb,tex,tsim
tmap = assignTextons(fbRun(textonData.fb,im),textonData.tex);
[tg,theta] = tgmo(tmap,k,idiag*radius,norient,...
                  'smooth','savgol','sigma',idiag*radius);
              
              
    function [cg,theta] = detCG(im,radius,norient)
        % function [cg,theta] = detCG(im,radius,norient)
        % Compute smoothed but not thinned CG fields.
        
         radius=0.02; norient=8; 
        
        [h,w,unused] = size(im);
        idiag = norm([h w]);
        
        % compute color gradient
        [cg,theta] = cgmo(im,idiag*radius,norient,...
            'smooth','savgol','sigmaSmo',idiag*radius);
        if (ndims(cg) == 4), cg = permute(cg,[1,2,4,3]); end
