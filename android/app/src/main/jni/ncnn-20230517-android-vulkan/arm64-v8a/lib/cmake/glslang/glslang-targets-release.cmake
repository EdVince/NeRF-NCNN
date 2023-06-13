#----------------------------------------------------------------
# Generated CMake target import file for configuration "release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "glslang::OSDependent" for configuration "release"
set_property(TARGET glslang::OSDependent APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::OSDependent PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libOSDependent.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::OSDependent )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::OSDependent "${_IMPORT_PREFIX}/lib/libOSDependent.a" )

# Import target "glslang::glslang" for configuration "release"
set_property(TARGET glslang::glslang APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::glslang PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libglslang.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::glslang )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::glslang "${_IMPORT_PREFIX}/lib/libglslang.a" )

# Import target "glslang::MachineIndependent" for configuration "release"
set_property(TARGET glslang::MachineIndependent APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::MachineIndependent PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMachineIndependent.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::MachineIndependent )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::MachineIndependent "${_IMPORT_PREFIX}/lib/libMachineIndependent.a" )

# Import target "glslang::GenericCodeGen" for configuration "release"
set_property(TARGET glslang::GenericCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::GenericCodeGen PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGenericCodeGen.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::GenericCodeGen )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::GenericCodeGen "${_IMPORT_PREFIX}/lib/libGenericCodeGen.a" )

# Import target "glslang::OGLCompiler" for configuration "release"
set_property(TARGET glslang::OGLCompiler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::OGLCompiler PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libOGLCompiler.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::OGLCompiler )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::OGLCompiler "${_IMPORT_PREFIX}/lib/libOGLCompiler.a" )

# Import target "glslang::SPIRV" for configuration "release"
set_property(TARGET glslang::SPIRV APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(glslang::SPIRV PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libSPIRV.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS glslang::SPIRV )
list(APPEND _IMPORT_CHECK_FILES_FOR_glslang::SPIRV "${_IMPORT_PREFIX}/lib/libSPIRV.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
