# SDK layout

Start with [START.md](../START.md). Run commands from the SDK root.

| Directory | Contents |
| --- | --- |
| `exercises/` | Programs and student blocks you edit. |
| `include/`, `lib/`, `python/` | Public APIs and compiled libraries. |
| `bin/` | Simulator/viewer and its ROS bridge. |
| `data/` | Reference paths. |
| `config/` | Simulator model settings. SLAM defaults are compiled in. |
| `docs/` | Guides, dependencies, build information and licenses. |

The autopilot drives on your pose filter, path tracker and low-level
controller. CogniCar owns live SLAM internally.

- [Getting started](getting-started-handout.pdf)
- [Path tracking](pathtracking-handout.pdf)
- [C++, Python and ROS 2 API](cognicar-api.pdf)
- [Real-car localization](LOCALIZATION.md)
- [ROS topics](TOPICS.md)
- [Dependencies](REQUIREMENTS.md), [build information](BUILD_INFO.txt), [license](LICENSE.txt)

`Latest*` getters reject missing or stale feedback. The default deadline is
250 ms. With real-car SLAM enabled, `LatestModel()` also requires an accepted
correction within 500 ms. See the API guide for freshness and units.

Qt notices are in `third_party/qt/` when the simulator is included.
