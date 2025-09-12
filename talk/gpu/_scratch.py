

# def _is_in_clip(a: np.ndarray, b: np.ndarray, c: np.ndarray, depth_buffer: np.ndarray) -> bool:
#     return (
#         -a[3] <= a[0] <= a[3] and -a[3] <= a[1] <= a[3] and a[2] > 0 and
#         -b[3] <= b[0] <= b[3] and -b[3] <= b[1] <= b[3] and b[2] > 0 and
#         -c[3] <= c[0] <= c[3] and -c[3] <= c[1] <= c[3] and c[2] > 0 and
#         a[2] <= depth_buffer[int(np.round(a[0])), int(np.round(a[1]))] and
#         b[2] <= depth_buffer[int(np.round(b[0])), int(np.round(b[1]))] and
#         c[2] <= depth_buffer[int(np.round(c[0])), int(np.round(c[1]))]
#     )