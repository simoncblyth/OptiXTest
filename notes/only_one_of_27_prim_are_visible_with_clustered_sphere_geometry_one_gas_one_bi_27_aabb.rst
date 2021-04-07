only_one_of_27_prim_are_visible_with_clustered_sphere_geometry_one_gas_one_bi_27_aabb
=======================================================================================


issue FIXED : it was a trivial bug of accidentally conflating transform_idx and idx in Foundry::makeClustered
----------------------------------------------------------------------------------------------------------------

* the result of that bug was for the SBTIndexOffset to be off by one, which means that one of the SBT records
  was being read from beyond the allocation

* given that bug its curious the outcome was that only the last sphere to appear   

::

    418 Solid* Foundry::makeClustered(const char* name,  int i0, int i1, int is, int j0, int j1, int js, int k0, int k1, int ks, double unit, bool inbox )
    419 {
    420 
    ...
    450     Solid* so = addSolid(numPrim, name);
    451     unsigned idx = 0 ;
    452 
    453     AABB bb = {} ;
    454 
    455     for(int i=i0 ; i < i1 ; i+=is )
    456     for(int j=j0 ; j < j1 ; j+=js )
    457     for(int k=k0 ; k < k1 ; k+=ks )
    458     {
    459         unsigned numNode = 1 ;
    460         Prim* p = addPrim(numNode);
    461         Node* n = addNode(Node::Make(name)) ;
    462 
    463         const Tran<double>* translate = Tran<double>::make_translate( double(i)*unit, double(j)*unit, double(k)*unit );
    464         unsigned transform_idx = 1 + addTran(*translate);      // 1-based idx, 0 meaning None
    465         n->setTransform(transform_idx);
    466         const qat4* t = getTran(transform_idx-1u) ;
    467 
    468         t->transform_aabb_inplace( n->AABB() );
    469 
    470         bb.include_aabb( n->AABB() );
    471 
    472         p->setSbtIndexOffset(idx) ;
    473         p->setAABB( n->AABB() );
    474 
    475         DumpAABB("p->AABB() aft setup", p->AABB() );
    476 
    477         std::cout << " idx " << idx << " transform_idx " << transform_idx << std::endl ;
    478 
    479         idx += 1 ;
    480     }
     


Issue
----------


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













