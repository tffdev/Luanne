# Luanne
a series of small apps that accommodate a very simple image classification machine learning library.

This does not have practical use outside of the research project, but if you'd like to play with/modify it, feel free!

### Dependencies:
* luafilesystem `sudo luarocks install luafilesystem`

### Limitations 
* Currently can only convert BITMAP images to vector files

# Running the Project

## Preparing your image folder to be vectorised
Run this command in your image folder, make sure to resize the images to 30x30
and replacing `*.jpg` with whatever filetype the images are currently 
```mogrify -format bmp -resize 30x30! *.jpg```

## Initialise project
navigate to the "src" folder and run:
```lua init.lua```
There will be on-screen instructions to follow, name your project something simple (with no spaces or dashes etc)


## Learning
Once you've initialised your new project, run:
```luajit main.lua learn {name of your project}```
to learn from your given image datasets.

![](http://tfcat.me/files/luanneimages/screencap2.png)

***NOTE***: To safely exit the learning loop (as to not cancel while saving synapses), change the value in "status" to `0`, and the process will be cancelled on the next synapse save. 


## Testing
To test a single vector file, run:
```luajit main.lua do {name of your project} {name of the file}```

![](http://tfcat.me/files/luanneimages/screencap1.png)