import open3d as o3d
import geometry3d

# 加载点云
pcd = o3d.io.read_point_cloud("/home/wangzy/lyf/ginger_ws/output/vis_hand.ply")

pcd = geometry3d.points_filter(pcd)
# pcd = pcd.cluster_dbscan(0.02, 3)
o3d.visualization.draw_geometries([pcd], window_name="点云凸包",
                                  width=800,  # 窗口宽度
                                  height=600)  # 窗口高度

# 点云AABB包围盒
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
# 点云OBB包围盒
obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)
# 可视化滤波结果
o3d.visualization.draw_geometries([pcd, aabb, obb], window_name="点云包围盒",
                                  width=800,  # 窗口宽度
                                  height=600)  # 窗口高度


# 计算凸包
hull, _ = pcd.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 1))
# 可视化滤波结果
o3d.visualization.draw_geometries([pcd, hull_ls], window_name="点云凸包",
                                  width=800,  # 窗口宽度
                                  height=600)  # 窗口高度