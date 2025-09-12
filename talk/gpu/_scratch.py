

# def draw_triangle(
#         a: np.ndarray, b: np.ndarray, c: np.ndarray, colors: np.ndarray,
#         xab_full: Optional[np.ndarray], xbc_full: Optional[np.ndarray],
#         xac_full: Optional[np.ndarray], xabc_full: Optional[np.ndarray],
#         bary: Optional[np.ndarray]):



# if should_do_allocation:
#     xab_full, xab_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
#     xbc_full, xbc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
#     xac_full, xac_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
#     xabc_full, xabc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
#     bary, bary_h = mem.get_sb(3, mem.SbType.FLOAT)



# if should_do_allocation:
#     mem.free_sb(bary_h, mem.SbType.FLOAT)
#     mem.free_sb(xabc_full_h, mem.SbType.INT)
#     mem.free_sb(xac_full_h, mem.SbType.INT)
#     mem.free_sb(xbc_full_h, mem.SbType.INT)
#     mem.free_sb(xab_full_h, mem.SbType.INT)











# xab_full, xab_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
# xbc_full, xbc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
# xac_full, xac_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
# xabc_full, xabc_full_h = mem.get_sb(SCRATCH_BUFFER_SIZE, mem.SbType.INT)
# bary, bary_h = mem.get_sb(3, mem.SbType.FLOAT)

# # TODO: Figure out how to broadcast this properly
# for (a, b, c), tcolors in zip(triangles, triangle_colors):
#     ap = transform @ a
#     bp = transform @ b
#     cp = transform @ c

#     if rasterizer.is_in_clip(ap, bp, cp):
#         num_tris += 1
#         direct.draw_triangle(
#             rasterizer.to_screen(ap), rasterizer.to_screen(bp), 
#             rasterizer.to_screen(cp), tcolors, 
#             xab_full, xbc_full, xac_full,
#             xabc_full, bary)
        
# mem.free_sb(bary_h, mem.SbType.FLOAT)
# mem.free_sb(xabc_full_h, mem.SbType.INT)
# mem.free_sb(xac_full_h, mem.SbType.INT)
# mem.free_sb(xbc_full_h, mem.SbType.INT)
# mem.free_sb(xab_full_h, mem.SbType.INT)