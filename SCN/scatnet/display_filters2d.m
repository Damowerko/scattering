function [ ] = display_filters2d( filters )
figure;
imagesc(filters.phi);
title('Phi');

step = 2*pi/size(filters.psi, 4);
thetas = 0:step:2*pi-step;

for j = 0:size(filters.psi, 3)-1
    for l = 1:size(filters.psi, 4);
        figure;
        imagesc(filters.psi(:,:,j+1,l));
        title(['j=' int2str(j) ' theta=' int2str(thetas(l))]);
    end
end

end

