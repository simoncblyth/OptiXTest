only_one_of_27_prim_are_visible_with_clustered_sphere_geometry_one_gas_one_bi_27_aabb
=======================================================================================


With env.sh::

    geometry=clustered_sphere
    clusterspec=-1:2:1,-1:2:1,-1:2:1
    clusterunit=500

    # default sphere is radius 100

::

    cd tests
    ./posi.sh 

    plot3d( posi[:,:3] )  # only one sphere shows in 3d plot : looks like center 500,500,500

Confirm that from radius for all intersects::

    p = hposi[:,:3] - np.array([500,500,500], dtype=np.float32)   

    In [6]: np.sqrt( np.sum( p*p, axis=1) )                                                                                                                                                                  
    Out[6]: array([100.   ,  99.999, 100.   , ..., 100.002, 100.002, 100.002], dtype=float32)

    In [7]: np.sqrt( np.sum( p*p, axis=1) ).min()                                                                                                                                                            
    Out[7]: 99.99707

    In [8]: np.sqrt( np.sum( p*p, axis=1) ).max()                                                                                                                                                            
    Out[8]: 100.003006


From logging output, that sphere is the last Prim of the cluster::

    primIdx  26 Prim mn (       400,       400,       400)  mx (       600,       600,       600)  sbtIndexOffset 27 numNode   1 nodeOffset  26 tranOffset   0 planOffset   0
    Node sphere        400        400        400        600        600        600 



Try using a small *clusterunit* so spheres are overlapping::

    clusterspec=-1:2:1,-1:2:1,-1:2:1
    clusterunit=100

Can see the "sheets" from three spheres. The view is tight though.













