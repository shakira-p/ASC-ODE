# ASC-ODE
A package for solving ordinary differential equations

Read the [documentation](https://tuwien-asc.github.io/ASC-ODE/intro.html)

Find theory behind here: https://jschoeberl.github.io/IntroSC/ODEs/ODEs.html

# get the program running wuhu

do this in src folder:

```
mkdir build
cd build
cmake ..
make
```

this is what I needed to install:
```

sudo apt install python3.12-ven
sudo  apt install python3-pybind11

```
and I also needed to install cmake, but I think only one of them worked. Probaly the second one:
```
sudo apt install cmake  # version 4.1.2
sudo apt  install cmake  # version 3.27.8-1build
```

then itshould run with the first block of code

That worked thanks! :D