# setup.py
import sys
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# 一个自定义的 build_ext 类，用于调用 CMake
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]
        
        build_args = ['--config', 'Release']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMake configure and build
        import subprocess
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='flash_join',
    version='0.1.0',
    author='Jinming Hu',
    author_email='conanhujinming@gmail.com',
    description='A high-performance hash join module',
    long_description='',
    ext_modules=[CMakeExtension('flash_join')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)