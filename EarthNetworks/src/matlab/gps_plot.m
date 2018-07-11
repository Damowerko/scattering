[G,S] = gps(0.6, false, 0.5);
n = length(S);
snsr_pos=rand(n,2); % Sensor positions

line_width = 1;
penn_blue = [0 71 133]/255;
penn_red = [169 5 51]/255;
marker_size = 2;

%% Plot graph
triuS=triu(S); % Because the graph is undirected I only care about
    % either the upper or lower triangular matrix of the adjacency
    % matrix.
[row_edges,col_edges]=find(triuS>0); % Location of edges in
    % the matrix
n_edges=length(row_edges); % Number of edges
edges_x=zeros(n_edges,2); % Each row of this matrix denotes the
    % x-coordinate of the start point and end point of each edge
edges_y=zeros(n_edges,2); % Each row of this matrix denotes the
    % y-coordinate of the start point and end point of each edge

for it=1:n_edges
    edges_x(it,1)=snsr_pos(row_edges(it),1);
    edges_x(it,2)=snsr_pos(col_edges(it),1);
    edges_y(it,1)=snsr_pos(row_edges(it),2);
    edges_y(it,2)=snsr_pos(col_edges(it),2);
end

%\\\ FIGURE
fig_graph=figure();
hold on
for it=1:n_edges
    plot_graph_edges=plot(edges_x(it,:),edges_y(it,:));
    set(plot_graph_edges,'LineWidth',0.5*line_width,...
        'color',penn_blue);
end
plot_graph_nodes=plot(snsr_pos(:,1),snsr_pos(:,2));
set(plot_graph_nodes,'LineStyle','none',...
    'MarkerSize',2*marker_size,'Marker','s',...
    'MarkerFaceColor',penn_blue,'MarkerEdgeColor',penn_blue);
plot_graph_ell_node=plot(snsr_pos(:,1),snsr_pos(:,2));
set(plot_graph_ell_node,'LineStyle','none',...
    'MarkerSize',3*marker_size,'Marker','s',...
    'MarkerFaceColor',penn_red,'MarkerEdgeColor',penn_red);
hold off
title('Sensor graph','Interpreter','LaTeX');
xlabel('$r_{1}$','Interpreter','LaTeX');
ylabel('$r_{2}$','Interpreter','LaTeX');
axis equal; axis([0 1 0 1]);
printpdf('gmrf_glln_01-graph','no-save');
%\\\