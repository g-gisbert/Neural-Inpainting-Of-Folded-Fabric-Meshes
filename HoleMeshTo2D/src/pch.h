#ifndef PCH_H
#define PCH_H

// Utilities
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <functional>
#include <algorithm>
#include <utility>
#include <tuple>
#include <string>
#include <cstring>
#include <chrono>
#include <fstream>
#include <limits>
#include <optional>

// STD Containers
#include <queue>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <deque>

// Geometry Central
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/pointcloud/point_cloud.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// Polyscope
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"

// Eigen
#include <Eigen/Core>

// LibIgl
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/lscm.h>
#include <igl/boundary_loop.h>

#include <igl/arap.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

// System
#include <dirent.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <sys/stat.h>

// Misc
#include <omp.h>
#include <random>
#include <stdlib.h>


#endif //PCH_H
