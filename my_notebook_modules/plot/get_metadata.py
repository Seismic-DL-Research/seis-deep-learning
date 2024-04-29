def get_metadata(data_elem__):
  str_info = [f'avg_ratio: {data_elem__["aavg_ratio"]:.4f}',
              f'overall_ratio: {data_elem__["overall_ratio"]:.4f}',
              f'magn: {data_elem__["magn"]:.2f}',
              f'evla: {data_elem__["evla"]:.3f}\U000000B0',
              f'evlo: {data_elem__["evlo"]:.3f}\U000000B0',
              f'stla: {data_elem__["stla"]:.3f}\U000000B0',
              f'stlo: {data_elem__["stlo"]:.3f}\U000000B0',
              f'dist: {data_elem__["dist"]:.2f} km',
              f'filename: {data_elem__["filename"].numpy().decode("utf-8")}',
              f'start: {data_elem__["start"].numpy().decode("utf-8")}']
  
  return str_info