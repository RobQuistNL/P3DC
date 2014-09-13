Installation Instructions:
GM8.X:
Import Resource P3DC.gmres, and include the dll in your games directory.

-------------------------------------------
-------------------------------------------
Version History
-------------------------------------------
-------------------------------------------


-------------------------------------------
Updates From V6.00 to 6.1
Fixed a possible bug with crashing when adding triangles sometimes.


-------------------------------------------
Updates From V5.02 to 6.00
-Support for all versions of gamemaker greater than 8.0 (inclusive)
-Faster raycasting functions (again!)
-Overwriting functions, allowing you to change the models in realtime without clearing and re-defining everything.
-Removed the direction arguments from the scripts due to exceeding the limit of 16 arguments per function. They were allowed before due to GMAPI, but since I have removed it's dependancy you now have to call additional functions.
-Call p3dc_set_modrot(<...>) to change a models rotation when being added


-------------------------------------------

Updates From V5.01 to 5.02
-Re added support for gamemaker primitives inside of external models. This includes all of them except the ellipsoid (Block, Floor, Wall, Cone, Cylinder).
-Fixed a confusing comment dealing with the direction of the ray.

-------------------------------------------

Updates From V5.00 to 5.01
-Fixed a bug with surface masks and rotated raycasting
-Fixed a bug with incorrect normals on cubes

-------------------------------------------

Updates From V4.00 to 5.00

-Surfaces can now have masks, so it is possible to create harmful surfaces, and detect what you're standing on. This can also be used to detect what you've just shot without using objects, where you shot the enemy (head, torso, arms, legs, etc). It makes your life easier and your game faster.

-------------------------------------------

Updates From V3.00 to 4.00
-Anywhere from 5%-50+% speed boost. If calling ALOT of small functions (IE: raycast on a cube) you'll see a huge speed improvement from 3.00. If you're calling a few raycasts on a big level the speed improvements are much less noticable.

-Uses GMAPI, so there is no dll overhead when calling the functions
-Rotation Support
-Rotate imported models (.d3d)
-Importing D3D models no longer supports gamemakers default primitives HOWEVER it still supports triagnlestrips, lists and fans.
-Rewrote all the comments in the scripts. They are now easier to understand
-Changed normals to be pre-calculated to improve framerates
-Updated all examples to better show the power of P3DC.dll
-Removed Example6. Editor14 used P3DC V2.00, and V4.00 is currently NOT compatible
with it. I'm going to update it soon though -> to use P3DC 4.00.

-------------------------------------------
Updates From V2.00 to V3.00

-Average 20% Faster doing the EXACT SAME THING vs. V2.00
-Added Normal Support, and ability to return the triangle cords hit
-Changed return 0 on no ray collision to return 10000000
-Model Splitting and Sorting, allowing for accelerated collisions
-Removed a few useless/non-working functions
-Made p3dc_check equal in speed to p3dc_check_still
-Made p3dc_ray equal in speed to p3dc_ray_still
-p3dc_ray_first is now useable on almost every map because of p3dc_sort_model
-p3dc_add_model has 3 new arguments, allowing you to place multiple of the same models on the map
-A few other [minor] updates


-------------------------------------------

All history previous to this has NOT been recorded.


Cheers
~~Brett Binnersley (Brett14);

--------------------- 
Edit from Javier Mora (Ciberman): 
I have adapted the examples to GM Studio. And thanks to Fishman1175 to find the solution for model rot bug.
