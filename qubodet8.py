import numpy as np
import neal

def qubo_det_8(terrain, heads, wet_cells, debug=False):
    """
    Deterministic-8 implementation using D-Wave Ocean SDK
    
    Takes 3 3x3 integer-valued grids which represent respectively:
    terrain: a 'Digital Elevation Model' (read from csv file with Geopandas)
    heads: the respective water heads for the terrain (as integers)
    wet_cells: binary grid of cells, 1 if they have water, 0 if not.
    Follows a one-hot scheme.

    Returns:
    Three 3x3 grids:
    The channel network
    The new configuration for the heads
    The new configuration for the wet_cells (binary grid)
    """
    bench = %timeit -n 1 -r 5  -o [x for x in range(10)] 
    print(f"Best time: {bench.best}")
    t_f, h_f, w_f = [x.flatten() for x in (terrain, heads, wet_cells)]

    aquifer_height = h_f[4] # Take central cell in original terrain 
    if (debug == True):
        print("aquifer initial height: ", aquifer_height.astype(int))
    # We analyze the terrain to develop the channel network
    neighbors_idx = [i for i in range(9)] #if i != 4]
    channel_network = (terrain - t_f[4]).astype(int)
    
    abs_diffs = {i: abs(t_f[i] - (h_f[4] + t_f[4])) for i in neighbors_idx} 
    P = max(abs_diffs.values()) * 2
    qubo = {}
    for i in neighbors_idx:
      qubo[(i, i)] = -abs_diffs[i] - P

    for i in neighbors_idx:
        for j in neighbors_idx:
            if i < j:
                qubo[(i, j)] = 2 * P

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=10) 
    best = response.first.sample
    
    destination = [j for j, v in best.items() if v == 1][0]
    # The next can be optimized defining new local variables, it is
    # written as it is for clarity
    if (abs_diffs[destination] - t_f[4] > h_f[4]):
        h_f[destination] = h_f[4]
        h_f[4] = 0
        w_f[4] = 0
        w_f[destination] = 1
    else:
        h_f[destination] = (h_f[4] - (t_f[destination] - t_f[4]))/2
        h_f[4] = h_f[4] - h_f[destination]
        w_f[destination] = 1
    if(debug == True):
        print("Sampler Properties:\n", sampler.properties)
        print("Sampler results:\n", response)
    
    return channel_network, h_f.reshape(terrain.shape), w_f.reshape(terrain.shape)

notation  = np.array([["a00","a01","a02"],["a10","a11","a12"],["a20","a21","a22"]])
terrain = np.array([[6, 5, 5], [7, 2, 5], [0, 4, 6]])
wet_cells = np.array([[0,0,0],[0,1,0],[0,0,0]])
heads = np.array([[0,0,0],[0,3,0],[0,0,0]])

print("\n----------------------------------------------------------------------------------------")
print("QUBO for deterministic-8\n(C) 2026, Jaime Anguiano Olarra (jaimeangola.github@gmail.com)")
print("Simple model for the widely used routine in Hydrology")
print("Read paper for more information (distributed under the GPLv3 (2007))")
print("----------------------------------------------------------------------------------------")
channel_network, newheads, newwets = qubo_det_8(terrain, heads, wet_cells, debug=True)
print(f"Notation, for a general matrix A:\n{notation}")
print(f"Terrain:\n{terrain}\nChannel network:\n{channel_network}\nHeads:\n{heads}\n")
print(f"Terrain:\n{terrain}\nNew heads:\n{newheads}\nNew wets:\n{newwets}\n")
