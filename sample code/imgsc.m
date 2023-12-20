function imgsc(u)

if size(u,3)>1
    u = uint8(u);
end

if ~isreal(u)
    u = abs(u);
end

% figure,
imagesc(u);
colormap(gray);
% colormap(jet);
axis('image');
axis off;

colorbar;
