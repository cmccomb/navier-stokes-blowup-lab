cmake_minimum_required(VERSION 3.24)
project(ns_static_refinement LANGUAGES C CXX)
include(FetchContent)

# Immutable upstream source snapshots, with archive integrity checks.
set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
set(AMReX_PRECISION DOUBLE CACHE STRING "" FORCE)
set(AMReX_MPI OFF CACHE BOOL "" FORCE)
set(AMReX_OMP OFF CACHE BOOL "" FORCE)
set(AMReX_FORTRAN OFF CACHE BOOL "" FORCE)
set(AMReX_PARTICLES OFF CACHE BOOL "" FORCE)
set(AMReX_AMRLEVEL OFF CACHE BOOL "" FORCE)
set(AMReX_EB OFF CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS ON CACHE BOOL "" FORCE)
set(AMReX_LINEAR_SOLVERS_EM OFF CACHE BOOL "" FORCE)
set(AMReX_GPU_BACKEND NONE CACHE STRING "" FORCE)
FetchContent_Declare(amrex
  URL https://codeload.github.com/AMReX-Codes/amrex/tar.gz/e60cdc18711ccf7fc0616d7a2fdf062021376976
  URL_HASH SHA256=8d8aa5990ad0f76326108084afcd2180cf9da431dc37fa99ad631318cd6a5fae)
FetchContent_MakeAvailable(amrex)

set(HYDRO_SPACEDIM 3 CACHE STRING "" FORCE)
set(HYDRO_MPI OFF CACHE BOOL "" FORCE)
set(HYDRO_OMP OFF CACHE BOOL "" FORCE)
set(HYDRO_EB OFF CACHE BOOL "" FORCE)
set(HYDRO_NO_EB ON CACHE BOOL "" FORCE)
set(HYDRO_PROJECTIONS ON CACHE BOOL "" FORCE)
FetchContent_Declare(hydro
  URL https://codeload.github.com/AMReX-Fluids/AMReX-Hydro/tar.gz/e49df248aabd2cc11865eb5be734a2f5f2f65ee5
  URL_HASH SHA256=00d847bc993508c80a42e746396d1401039360ed67d5201dc8a3d17225855a09)
FetchContent_MakeAvailable(hydro)

add_executable(ns_projection_pilot projection_pilot.cpp)
target_compile_features(ns_projection_pilot PRIVATE cxx_std_20)
target_link_libraries(ns_projection_pilot PRIVATE amrex_hydro)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/projection_pilot.cpp" pilot_source_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS projection_pilot.cpp)
target_compile_definitions(ns_projection_pilot PRIVATE
  NS_PILOT_SOURCE_SHA256="${pilot_source_sha256}"
  NS_AMREX_COMMIT="e60cdc18711ccf7fc0616d7a2fdf062021376976"
  NS_HYDRO_COMMIT="e49df248aabd2cc11865eb5be734a2f5f2f65ee5")
enable_testing()
add_test(NAME exact_rest COMMAND ns_projection_pilot n_cell=16 levels=3 test_case=rest)
add_test(NAME mixed_projection COMMAND ns_projection_pilot n_cell=16 levels=3 test_case=mixed)
add_test(NAME solenoidal COMMAND ns_projection_pilot n_cell=16 levels=4 test_case=solenoidal)
add_test(NAME four_level_projection COMMAND ns_projection_pilot n_cell=16 levels=4 test_case=mixed)

add_executable(ns_diffusion_pilot diffusion_pilot.cpp)
target_compile_features(ns_diffusion_pilot PRIVATE cxx_std_20)
target_link_libraries(ns_diffusion_pilot PRIVATE amrex_hydro)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/diffusion_pilot.cpp" diffusion_source_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS diffusion_pilot.cpp)
target_compile_definitions(ns_diffusion_pilot PRIVATE
  NS_DIFFUSION_SOURCE_SHA256="${diffusion_source_sha256}"
  NS_AMREX_COMMIT="e60cdc18711ccf7fc0616d7a2fdf062021376976")
add_test(NAME diffusion_rest COMMAND ns_diffusion_pilot test_case=rest steps=8)
add_test(NAME diffusion_constant COMMAND ns_diffusion_pilot test_case=constant steps=8)
add_test(NAME diffusion_decay COMMAND ns_diffusion_pilot test_case=decay steps=8)
