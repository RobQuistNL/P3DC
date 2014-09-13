#define p3dc_init
/*
P3DC (Precise 3D Collisions)
V6.00
----
CREDITS:

**Credit is required to the following people if used

Brett Binnersley (Brett14): Making P3DC.dll
Samuel Hanson (Hanson): Making ModModCollisions.dll
Snake_PL: Creating GMAPI, P3DC uses this to make calling all the functions faster
Thomas Moller: Some of ModMod Collisions Code was based off of his code (the math)

*/
n="P3DC.dll";

//**Model Creation
global.p3dc_bdm = external_define(n,"bdm",dll_cdecl,ty_real,0);
global.p3dc_edm = external_define(n,"edm",dll_cdecl,ty_real,0);
global.p3dc_apm = external_define(n,"apm",dll_cdecl,ty_real,4,ty_string,ty_real,ty_real,ty_real);
global.p3dc_stm = external_define(n,"stm",dll_cdecl,ty_real,1,ty_real);
global.p3dc_brm = external_define(n,"brm",dll_cdecl,ty_real,1,ty_real);
global.p3dc_erm = external_define(n,"erm",dll_cdecl,ty_real,0);
global.p3dc_bs3 = external_define(n,"bs3",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_mat = external_define(n,"mat_exported",dll_cdecl,ty_real,9,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_apw = external_define(n,"apw",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_apf = external_define(n,"apf",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_apb = external_define(n,"apb",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_apc = external_define(n,"apc",dll_cdecl,ty_real,8,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_apo = external_define(n,"apo",dll_cdecl,ty_real,8,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);

//Overwrite
global.p3dc_obd = external_define(n,"obd",dll_cdecl,ty_real,2,ty_real,ty_real);
global.p3dc_oed = external_define(n,"oed",dll_cdecl,ty_real,0);
global.p3dc_opt = external_define(n,"opt",dll_cdecl,ty_real,9,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_opw = external_define(n,"opw",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_opf = external_define(n,"opf",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_opb = external_define(n,"opb",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_opc = external_define(n,"opc",dll_cdecl,ty_real,8,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_opo = external_define(n,"opo",dll_cdecl,ty_real,8,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);

//**Return Values
global.p3dc_gmn = external_define(n,"gmn",dll_cdecl,ty_real,0);
global.p3dc_gmt = external_define(n,"gmt",dll_cdecl,ty_real,1,ty_real);
global.p3dc_gms = external_define(n,"gms",dll_cdecl,ty_real,2,ty_real,ty_real);
global.p3dc_gtr = external_define(n,"gtr",dll_cdecl,ty_real,1,ty_real);
global.p3dc_gtm = external_define(n,"gtm",dll_cdecl,ty_real,0);

//**Collision Detection
global.p3dc_mcs = external_define(n,"mcs",dll_cdecl,ty_real,8,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_mcr = external_define(n,"mcr",dll_cdecl,ty_real,11,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_mc3 = external_define(n,"mc3",dll_cdecl,ty_real,4,ty_real,ty_real,ty_real,ty_real);
global.p3dc_mrs = external_define(n,"mrs",dll_cdecl,ty_real,10,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_mrf = external_define(n,"mrf",dll_cdecl,ty_real,7,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);             
global.p3dc_mr3 = external_define(n,"mr3",dll_cdecl,ty_real,6,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);           
global.p3dc_mrr = external_define(n,"mrr",dll_cdecl,ty_real,10,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real,ty_real);
global.p3dc_smr = external_define(n,"smr",dll_cdecl,ty_real,3,ty_real,ty_real,ty_real);               
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  


#define p3dc_free
/*
P3DC (Precise 3D Collisions)
V6.00
----
Call this when the game ends.
*/
external_free("P3DC.dll");


#define p3dc_begin_model
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Call this before adding any polygons to a model, this creates a new collision model.
Call p3dc_end_model() when done adding polygons. You can only be editing one model
at a time

----
Returns the id of a new model
*/
return external_call(global.p3dc_bdm);

#define p3dc_end_model
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Call this when done adding any polygons to a model;
----
Returns the number of triangles just added
*/
return external_call(global.p3dc_edm);

#define p3dc_add_model
/* 
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Adds an external D3D model to the current model; Current model is returned by the last called p3dc_begin_model();
You usually want to sent The models X,Y,Z to 0, unless they are non moving objects such as trees. The model will
rotate itself around where you place it. So if you place it at 0,0,0 then the model will be rotated around [0,0,0]
The model gets rotated by the direction in a vector declared in p3dc_set_modrot(<...>);
        
----
RETURNS:
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.

UNLESS it is one of the following values...

1: model not found
2: Models version is not correct/incorrect formatting
3: Model contains no DATA
4: Failed opening the model file for reading (it was found)
5: Unsupported data is contained within the model (pointlist etc.)

----
ARGUMENTS:

Arg0: Path to the model
Arg1: Model X
Arg2: Model Y
Arg3: Model Z
For model rotation call p3dc_set_modrot(<...>);
*/
return external_call(global.p3dc_apm,argument0,argument1,argument2,argument3);


#define p3dc_set_trianglemask
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

This allows surfaces to have masks, or represent objects. You can give
triangles values (integers) and those integers represent different surfaces.
This can be used to make ponds of water/lava, or in a racing game to detect
when the player runs off the track and onto grass, or when he runs into spikes,
or where you shoot an enemy (head, torso, arm, leg, hand, foot) etc.

Every triangle/shape that you add AFTER this is called will have this ID until
it is changed again. Look at the example below.

Set this to "-1" to represent no surface

To be used with p3dc_get_lastmask

----
RETURNS:

Nothing
*/
return external_call(global.p3dc_stm,argument0);

#define p3dc_begin_replace
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Replacing models will delete the existing one, and can vary in size from before
it was cleared. Use the add functions with replacing.
Overwriting models can only modify existing data. It is faster and does not clear
any of the existing data, but may not change the size of the model.

----
ARGUMENTS:

Arg0: P3DC Model ID

----
returns nothing
*/
return external_call(global.p3dc_brm,argument0);

#define p3dc_end_replace
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Call this when done changing the collision polygons for a model;

----
RETURNS:

Returns true is successful

*/
return external_call(global.p3dc_erm);

#define p3dc_clear_model
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Call this to clear the polygons in a model, freeing the memory it uses.
Note that this doesn't actually DELETE the model, it just clears all the
data contained within it.

----
ARGUMENTS:

Arg0: P3DC Model ID

----
EXAMPLE:

p3dc_clear_model(BOSS_WE_WILL_NEVER_SEE_AGAIN_MODEL);
*/
p3dc_begin_replace(argument0);
p3dc_end_replace();


#define p3dc_begin_overwrite
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Overwriting models can only modify existing data. It can be used for parts
of the model that move (an elevator for example). Not every triangle needs
to be overwritten, you can choose to only overwrite parts of the model. An
elevator on a level for example.

YOU SHOULD NEVER overwrite a model that has been split. You will get undefined
behaviour if you do this.

----
Arg0: The model ID to start overwriting
Arg1: The Model TLID to start overwriting at
*/
return external_call(global.p3dc_obd,argument0,argument1);


#define p3dc_end_overwrite
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Call this when done overwriting a model
*/
return external_call(global.p3dc_oed);


#define p3dc_overwrite_triangle
/*
Overwrites a triangle in the model.

Arg0: Point1 X
Arg1: Point1 Y
Arg2: Point1 Z

Arg3: Point2 X
Arg4: Point2 Y
Arg5: Point2 Z

Arg6: Point3 X
Arg7: Point3 Y
Arg8: Point3 Z


Returns the TLID of the model.
This is rarely used from this point.
*/
return external_call(global.p3dc_opt,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8);

#define p3dc_overwrite_block
/*
Overwrites a block in the model.

Arg0: Point1 X
Arg1: Point1 Y
Arg2: Point1 Z

Arg3: Point2 X
Arg4: Point2 Y
Arg5: Point2 Z


Returns the TLID of the model.
This is rarely used from this point.
*/
return external_call(global.p3dc_opb,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_overwrite_floor
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Overwrites a floor in a model.

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

----
Returns the TLID of the model.
This is rarely used from this point.

*/
return external_call(global.p3dc_opf,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_overwrite_wall
/* 
P3DC (Precise 3D Collisions)
V6.00
----
Overwrites a wall in a model.

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

----
Returns the TLID of the model.
This is rarely used from this point.

*/
return external_call(global.p3dc_opw,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_overwrite_cylinder
/*
P3DC (Precise 3D Collisions)
V6.00
----
Overwrites a cylinder in a model. You should NEVER change the amount of
steps around it, or the closed argument. Maintain the same amount of
polygons in the cylinder.


Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

Arg6: Closed
Arg7: Steps

----
Returns the TLID of the model.
This is rarely used from this point.
*/
return external_call(global.p3dc_opc,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7);

#define p3dc_overwrite_cone
/*
P3DC (Precise 3D Collisions)
V6.00
----
Overwrites a cone in a model. You should NEVER change the amount of
steps around it, or the closed argument. Maintain the same amount of
polygons in the cylinder.

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

Arg6: Closed
Arg7: Steps

----
RETURNS:
Returns the TLID of the model.
This is rarely used from this point.
*/
return external_call(global.p3dc_opo,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7);

#define p3dc_split_model
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

-Splits a model into many smaller ones to be collision checked. To be used with models that have an origin of [0,0,0].
-Only One model can be split at a time (Split the level)
-Make sure that WIDTH divided by X REGIONS turns out to be a whole number, or risk an unexpected error
-Make sure that HEIGHT divided by Y REGIONS turns out to be a whole number, or risk an unexpected error.
-Use the split collision checking for it to be optimized, and additionally you can return the collision model id of
part of the model with p3dc_get_splitid(...)

----
ARGUMENTS:

Arg0: Model to split
Arg1: Model Width (Positive whole number, make sure it's LARGER than the furthest away X value from [0,0,0])
Arg2: Model Height (Positive whole number, make sure it's LARGER than the furthest away Y value from [0,0,0])
Arg3: X Regions (The number of times to split the level across the X plane) - MAXIMUM OF 50 REGIONS
Arg4: Y Regions (The number of times to split the level across the Y plane) - MAXIMUM OF 50 REGIONS
Arg5: "Extra Space". This is the area/space that the regions will overlap each other with.

*/
return external_call(global.p3dc_bs3,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_add_triangle
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Add a polygon to the current model; Current model is returned by the last called begin_define_model();

----
ARGUMENTS:

Arg0: Point1 X
Arg1: Point1 Y
Arg2: Point1 Z

Arg3: Point2 X
Arg4: Point2 Y
Arg5: Point2 Z

Arg6: Point3 X
Arg7: Point3 Y
Arg8: Point3 Z

----
RETURNS:

Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.

*/
return external_call(global.p3dc_mat,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8);

#define p3dc_add_block
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Adds a block to the current model; Current model is returned by the last called p3d_begin_model();
Same arguments for d3d_draw_block(x1,y1,z1,x2,y2,z2);

----
ARGUMENTS:

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

----
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.

*/

return external_call(global.p3dc_apb,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_add_floor
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Adds a floor to the current model; Current model is returned by the last called p3dc_begin_model();
Same arguments for d3d_draw_floor(x1,y1,z1,x2,y2,z2);

----
ARGUMENTS:

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

----
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.

*/
return external_call(global.p3dc_apf,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_add_wall
/* 
P3DC (Precise 3D Collisions)
V6.00
----
Adds a wall to the current model; Current model is returned by the last called p3dc_begin_model();
Same arguments for d3d_draw_wall(x1,y1,z1,x2,y2,z2);

----
ARGUMENTS:

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

----
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.

*/
return external_call(global.p3dc_apw,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_add_cylinder
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Adds a cylinder to the current model; Current model is returned by the last called p3dc_begin_model();
Same arguments for d3d_draw_cylinder(x1,y1,z1,x2,y2,z2,closed,steps);

----
ARGUMENTS:

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

Arg6: Closed
Arg7: Steps

----
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.
*/
return external_call(global.p3dc_apc,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7);

#define p3dc_add_cone
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Adds a cone to the current model; Current model is returned by the last called p3dc_begin_model();
Same arguments for d3d_draw_cone(x1,y1,z1,x2,y2,z2,closed,steps);

----
ARGUMENTS:

Arg0: X1
Arg1: Y1
Arg2: Z1

Arg3: X2
Arg4: Y2
Arg5: Z2

Arg6: Closed
Arg7: Steps

----
RETURNS:
Returns the triangle location identifier (Triangle LID).
Only used for overwriting models after they've been created.
*/
return external_call(global.p3dc_apo,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7);

#define p3dc_get_triangles
/*
P3DC (Precise 3D Collisions)
V6.00
----
ARGUMENTS:

Arg0: Model ID
----
Returns the number of polygons (triangles) in the model <Argument0>

*/
return external_call(global.p3dc_gmt,argument0);

#define p3dc_get_models
/*
P3DC (Precise 3D Collisions)
V6.00
----
Returns the total number of models, that is the number if times begin_define_model() has been called

*/
return external_call(global.p3dc_gmn);

#define p3dc_get_splitid
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

No error checking so picking a spot not in the included region (defined in p3dc_split_model) will cause an unexpected error
This can be used to check for rotated checks on the split level (rotated player) - look at the example below to see how

----
ARGUMENTS:

Arg0: X
Arg1: Y

----
Returns the id of the split model at X,Y 
*/
return external_call(global.p3dc_gms,argument0,argument1);

#define p3dc_get_lastmask
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES: To be used with p3dc_set_trianglemask

----
Returns the mask of the last triangle hit (by a ray, using one of the raycasting functions)

*/
return external_call(global.p3dc_gtm);

#define p3dc_triangle_data
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Returns a value from the last triangle hit
No error checking, so picking anything below 0 or above 12 will lead to an unexpected error

----
ARGUMENTS:

Arg0: Set this to a value between 0 and 12 -> 0:X1, 1:Y1, 2:Z1, 3:X2, 4:Y2, 5:Z2, 6:X3, 7:Y3, 8:Z3, 9:NX, 10:NY, 11:NZ, 12:Triangle ID

NX,NY,NZ are the normals of the triangle
Triangle ID is the triangle number (if it was the 5th triangle added, it'll return 5).
TriangleID will NOT be correct if you sort the model, because it moves all the polygons around.
The first 9 numbers (0-8) represent the vertexes of the triangle

*/
return external_call(global.p3dc_gtr,argument0);

#define p3dc_check
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between two 3d models; Both of which can move.
The model ID's are the ones returned from p3dc_begin_model() - ARGUMENTS SECTION

----
RETURNS:

returns (1): there is a collision
returns (0): no collision
returns (any other): Error. This is often caused by an incorrect amount
of vertexes/polygons added to the model (say 4 vertexes on a trianglelist)

----
ARGUMENTS:

Arg0: model1 id
Arg1: model1 x
Arg2: model1 y
Arg3: model1 z
Arg4: model2 id
Arg5: model2 x
Arg6: model2 y
Arg7: model2 z

*/
return external_call(global.p3dc_mcs,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7);

#define p3dc_check_still
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between two 3d models; Model two can not move. If the model is static it is HIGHLY
recommended to add it to the level model that gets split. Split functions preform a lot faster than a still
check, and they'll do the same thing.

----
ARGUMENTS:

Arg0: model1 id
Arg1: model1 x
Arg2: model1 y
Arg3: model1 z
Arg4: model2 id

*/
return p3dc_check(argument0,argument1,argument2,argument3,argument4,0,0,0);

#define p3dc_check_rotation
/*
P3DC (Precise 3D Collisions)
V6.00
----
Check for a collision between two 3d models; Both of which can move.

----
RETURNS:

returns (1): there is a collision
returns (0): no collision
returns (any other): Error

----
ARGUMENTS:

Arg0: model1 id
Arg1: model1 x
Arg2: model1 y
Arg3: model1 z
Arg4: model2 id
Arg5: model2 x
Arg6: model2 y
Arg7: model2 z
[Arg8-10] Creates a vector that represent the model 1's rotation
Arg8: Vector X component
Arg9: Vector Y component
Arg10: Vector Z component
...For the vector that represents model2's rotation use p3dc_set_modrot(<...>);

*/
return external_call(global.p3dc_mcr,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8,argument9,argument10);

#define p3dc_check_split
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between a 3d model and the split model, which
SHOULD be the level model.

If the model [X/Y] origin are not in the space defined with p3dc_split_model, then it will cause an unexpected error

----
ARGUMENTS:

Arg0: model1 id
Arg1: model1 x
Arg2: model1 y
Arg3: model1 z

----
RETURNS:
1: if there is a collision
0: there is no collision
<Any other>: ERROR

*/
return external_call(global.p3dc_mc3,argument0,argument1,argument2,argument3);

#define p3dc_ray
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between a ray(3d vector) and a 3d model.

----
ARGUMENTS:

Arg0: model ID
Arg1: model X
Arg2: model Y
Arg3: model Z
Arg4: Ray x origin
Arg5: Ray y origin
Arg6: Ray y origin
[Arg7-9] Creates a vector that represent the rays direction
Arg7: Vector X component
Arg8: Vector Y component
Arg9: Vector Z component

----
The distance to the *closest* triangle that was hit. Returns 10000000 if no triangle was hit.
*/
return external_call(global.p3dc_mrs,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8,argument9);

#define p3dc_ray_still
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between a ray(3d vector) and a 3d model that doesn't move.

----
ARGUMENTS:

Arg0: model ID
Arg1: Ray x origin
Arg2: Ray y origin
Arg3: Ray y origin
[Arg4-6] Creates a vector that represent the rays direction
Arg4: Vector X component
Arg5: Vector Y component
Arg6: Vector Z component

----
Returns the distance to the *closest* triangle that was hit. Returns 10000000 if no triangle was hit.
*/
return p3dc_ray(argument0,0,0,0,argument1,argument2,argument3,argument4,argument5,argument6);

#define p3dc_ray_first
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

*Make sure you call p3dc_sort_model(); before using this. It is the fastest ray check available in p3dc
*This will not always work if you have triangles that stick through each other
*Check for a collision between a ray(3d vector) and a 3d model, 3D model does not move (read below to see how
to use it on models that move)

*This can be used to simply check if 2 objects can see each other, or simply if when you shoot you hit a
bot.

----
ARGUMENTS:

Argument0 - model id
Argument1 - Ray x origin
Argument2 - Ray y origin
Argument3 - Ray y origin
[Arg4-6] Creates a vector that represent the rays direction
Argument4 - Vector X component
Argument5 - Vector Y component
Argument6 - Vector Z component

----
RETURNS:

The distance to the *first* triangle that was hit, not necessarily the closest. Returns 10000000 if no triangle was hit.

*/
return external_call(global.p3dc_mrf,argument0,argument1,argument2,argument3,argument4,argument5,argument6);

#define p3dc_ray_split
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

This will check for a ray collision on the 3D model that has been split. The split
model should be the level. This is the *FASTEST* function in P3DC, because splitting
the model and calling the split functions uses an accelerated form of a QUADTREE for
collisions. This minimizes the ammount of triangles that need to be checked, up to 2500x
less. Recommended to use this function on the level


If the ray [X/Y] origin are not in the space defined with p3dc_split_model, then it will cause an unexpected error


----
ARGUMENTS:

Argument0: Ray x origin
Argument1: Ray y origin
Argument2: Ray y origin
[Arg3-5] Creates a vector that represent the rays direction
Argument3: Vector X component
Argument4: Vector Y component
Argument5: Vector Z component

----
Returns The distance to the *closest* triangle that was hit. Returns 10000000 if
no triangle was hit.
*/

return external_call(global.p3dc_mr3,argument0,argument1,argument2,argument3,argument4,argument5);

#define p3dc_ray_rotation
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Check for a collision between a ray(3d vector) and a 3d model that can rotate

----
ARGUMENTS:

Arg0: model ID
Arg1: model X
Arg2: model Y
Arg3: model Z
Arg4: Ray x origin
Arg5: Ray y origin
Arg6: Ray y origin
[Arg7-9]: Creates a vector that represent the rays direction
Arg7: Vector X component
Arg8: Vector Y component
Arg9: Vector Z component
...For the vector that represents the models rotation
use p3dc_set_modrot(<...>);
----
Returns the distance to the *closest* triangle that was hit. Returns 10000000 if no triangle was hit.
*/
return external_call(global.p3dc_mrs,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8,argument9);

#define p3dc_set_modrot
/*
P3DC (Precise 3D Collisions)
V6.00
----
NOTES:

Sets the rotation of Model (1 or 2) to be used with:
p3dc_check_rotation(<...>); and p3dc_ray_rotation(<...>);

----


[Arg0-2]: Creates a vector that represent the models rotation
Arg0: Vector X component
Arg1: Vector Y component
Arg2: Vector Z component

----
Returns true
*/
//return external_call(global.p3dc_smr,argument0,argument1,argument2);
return external_call(global.p3dc_mrr,argument0,argument1,argument2,argument3,argument4,argument5,argument6,argument7,argument8,argument9);

