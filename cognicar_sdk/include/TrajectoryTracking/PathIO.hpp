// Dependency-free path loading helpers.
//
// Supports the two formats used in this project: NumPy .npy (so the existing
// oval_path_*.npy files work directly) and CSV/whitespace text. All loaders
// return a normalized Nx4 path (x, y, heading, speed); heading is derived from
// the geometry when the file only has x, y.
#ifndef COGNI_TRACKING_PATH_IO_HPP
#define COGNI_TRACKING_PATH_IO_HPP

#include <string>
#include <vector>

#include <CogniCommon/Export.hpp>  // TRAJECTORY_TRACKING_API
#include <TrajectoryTracking/Types.hpp>

namespace cogni::tracking {

// Turn raw rows (each: x, y[, heading[, speed]]) into a normalized path.
// When a row has fewer than 3 columns, heading is derived from consecutive
// points; speed defaults to 0 when absent.
TRAJECTORY_TRACKING_API std::vector<PathPoint> normalizePath(const std::vector<std::vector<double>>& rows);

// Load a NumPy .npy array (C-order, 2D, float32 or float64).
TRAJECTORY_TRACKING_API std::vector<PathPoint> loadPathNpy(const std::string& file);

// Load a CSV / whitespace-separated text file. Lines starting with '#' and
// blank lines are ignored.
TRAJECTORY_TRACKING_API std::vector<PathPoint> loadPathCsv(const std::string& file);

// Dispatch on file extension (.npy -> loadPathNpy, otherwise CSV/text).
TRAJECTORY_TRACKING_API std::vector<PathPoint> loadPath(const std::string& file);

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_PATH_IO_HPP
