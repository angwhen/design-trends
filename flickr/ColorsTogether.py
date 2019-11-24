import YearlyColorGrids, ClusterImagesByColor

#ClusterImagesByColor.make_color_clusters(Q=5, K=7, color_rep="hsv",remove_monochrome=True, remove_heads = True, remove_skin=True)
YearlyColorGrids.yearly_grids(num_dom_colors=10, Q= 25, color_rep="hsv",remove_monochrome=True, remove_heads = True, remove_skin=True)
