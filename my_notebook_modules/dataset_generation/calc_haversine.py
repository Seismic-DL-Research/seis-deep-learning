import math

def calc_haversine(p1__, p2__):
  lat1, lon1 = p1__
  lat2, lon2 = p2__

  lat1 = math.radians(lat1)
  lat2 = math.radians(lat2)
  diffLon = math.radians(lon2 - lon1)

  a = math.sin(lat1) * math.sin(lat2)
  b = math.cos(lat1) * math.cos(lat2) * math.cos(diffLon)
  centralAngle = math.acos(a + b)

  dist = 6371.0088 * centralAngle
  return dist