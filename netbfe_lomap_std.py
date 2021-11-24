import os

import numpy as np
import networkx as nx
from scipy import linalg
from cvxopt import matrix

from DiffNet.diffnet import covariance
from DiffNet.diffnet import A_optimize
from DiffNet.diffnet import round_to_integers

def format_array(array, format_length):
    ret = []
    format_template = '%%-%ds'%(format_length+4)
    for item in array:
        ret.append(format_template % item)
    return ret

def print_array(array_name, array_content, round_n = 6):
    print('%s:'%array_name)
    array_content = np.array(array_content)
    max_element_length = -1
    shape = array_content.shape
    if len(shape) == 1:
        for i in range(shape[0]):
                if isinstance(array_content[i], float):
                    array_content[i] = round(array_content[i], round_n)
                max_element_length = max(max_element_length, len(str(array_content[i])))
        p_list = format_array(array_content, format_length=max_element_length)
        print('[ ' + ','.join(p_list) + ' ]')
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if isinstance(array_content[i][j], float):
                    array_content[i][j] = round(array_content[i][j], round_n)
                max_element_length = max(max_element_length, len(str(array_content[i][j])))
        print('[')
        for i in range(shape[0]):
            p_list = format_array(array_content[i], format_length=max_element_length)
            print('[ ' + ','.join(p_list) + ' ],')
        print(']')
    else:
        raise Exception('array shape must 1-dim or 2-dim')

def matrix_to_graph( A, origins='O'):
    '''
    Construct a graph of K+1 nodes from the KxK symmetric matrix A,
    where the weight of edge (i,j) is given by A[i][j], and the weight
    of edge (K=origin, i) is given by A[i][i].
    '''
    A = np.array( A)
    g = nx.from_numpy_matrix( A)
    if type(origins)==list:
        originIDs = list(set( origins))
        originIDs.sort()
        for o in originIDs:
            g.add_node( o)
    else:
        if origins is None: origins = 'O'
        g.add_node( origins)
        origins = [ origins ]*A.shape[0]
    for i in xrange( A.shape[0]):
        if A[i][i] != 0:
            g.remove_edge( i, i)
            g.add_edge( origins[i], i, weight=A[i][i])
    return g

def find_k_connected_sub_network(m, connectivity = 2, rbfe_only = True, additional = 0):
    """find k_connected sub-network with the minimum number 
        of edges and the largest allocations.
    Args:
        m (matrix): graph matrix.
        connectivity (int): connectivity number.
        rbfe_only (bool): If true, find the k connected graph without origin.
    Returns:
        if found, return edges of k connected sub network,
        otherwise, return None.
    """
    mtx = matrix(np.array(m))
    mtx = 1.0 * mtx / sum(mtx)
    K = mtx.size[0]
    g = matrix_to_graph(mtx, origins='O')
    edges=sorted(g.edges(data=True), key=lambda t: -t[2]['weight'])

    if rbfe_only:
        node_number = K
    else:
        node_number = K + 1

    additional_number = 0
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1], weight = e[2]["weight"])
        if rbfe_only and (e[0] == 'O' or e[1] == 'O'):
            print("Warning: without origin, but edge contain origin!")
        if len(G.nodes) == node_number and nx.is_k_edge_connected(G, k=connectivity):
            if additional_number < additional:
                additional_number += 1
            else:
                edges = set()
                for e in G.edges:
                    if e[1] == 'O':
                        edges.add((e[0], e[0]))
                    elif e[0] == 'O':
                        edges.add((e[1], e[1]))
                    else:
                        edges.add(e)
                return edges
    return None

def cutoff(nij, nmols, abfe_cutoff, rbfe_cutoff, abfe_lambdas, rbfe_lambdas):
    """Cutoff the small allocations in nij.

    Args:
        nij (matrix): Original nij matrix

    Returns:
        matrix: the cutoff matrix
    """
    for i in range(nmols):
        for j in range(nmols):
            if i == j and nij[i,j] < (abfe_cutoff * abfe_lambdas):
                nij[i][j] = 0
            if i != j and nij[i,j] < (rbfe_cutoff * rbfe_lambdas):
                nij[i][j] = 0
    return nij

def optimize(nsofar, sij, delta_N, ste, nmols, N_scale, abfe_cutoff, rbfe_cutoff, abfe_lambdas, rbfe_lambdas):

    for i in range(nmols):
        sij[i][i] = np.infty
    nij = A_optimize(matrix(sij), delta_N, matrix(nsofar), ste, method='conelp')
    nij = round_to_integers(nij)
    nij = cutoff(nij, nmols=nmols, abfe_cutoff=abfe_cutoff, rbfe_cutoff=rbfe_cutoff, abfe_lambdas=abfe_lambdas, rbfe_lambdas=rbfe_lambdas)
    for retry in range(10):
        k_connect_edges = find_k_connected_sub_network(nsofar + np.array(nij), connectivity=2, rbfe_only=True, additional=retry)
        if k_connect_edges:
            try:
                nij = A_optimize(matrix(sij), delta_N, matrix(nsofar),
                    ste, k_connect_edges, method='conelp')
                nij = round_to_integers(nij)
                nij = cutoff(nij, nmols=nmols, abfe_cutoff=abfe_cutoff, rbfe_cutoff=rbfe_cutoff, abfe_lambdas=abfe_lambdas, rbfe_lambdas=rbfe_lambdas)
                break
            except:
                print("fail stage-2 A_optimize, retry = %d" % retry)
    nij = np.array(nij)
    nsofar += nij
    nsofar /= N_scale
    return nsofar.tolist()

def cal_std(invv, delta_value):
    nmols = invv.shape[0]
    delta = np.array([np.infty for _ in range(nmols)])
    delta[0] = delta_value

    covar = covariance(invv, delta)
    std = [np.sqrt(covar[i][i]) for i in range(nmols)]

    return std

def optimal_std(nsofar, sij, delta_value, delta_N, ste, nmols, N_scale, abfe_cutoff, rbfe_cutoff, abfe_lambdas, rbfe_lambdas):
    next_nsofar = optimize(nsofar, sij, delta_N, ste, nmols, N_scale, abfe_cutoff, rbfe_cutoff, abfe_lambdas, rbfe_lambdas)
    invv = np.zeros((nmols, nmols), dtype=np.float)
    for i in range(nmols):
        for j in range(nmols):
            if not np.isclose(next_nsofar[i][j], 0):
                var = sij[i][j] * sij[i][j] / next_nsofar[i][j]
                invv[i][j] = 1./var

    std = cal_std(invv, delta_value=delta_value)
    return std

def lomap_std(nij, sij, delta_value):
    assert sij.shape == nij.shape
    nmols = sij.shape[0]

    invv = np.zeros((nmols, nmols), dtype=np.float)
    for i in range(nmols):
        for j in range(nmols):
            if not np.isclose(nij[i][j], 0):
                var = sij[i][j] * sij[i][j] / nij[i][j]
                invv[i][j] = 1./var

    std = cal_std(invv, delta_value=delta_value)
    return std

if __name__ == "__main__":
    # netbfe
    invv = np.array([
                [ 0.0        ,400.0      ,123.457    ,59.172     ,100.0      ,277.778    ,14.793     ,204.082    ,82.645     ,204.082    ,82.645     ,277.778    ,400.0      ,123.457    ,100.0      ,30.864      ],
                [ 400.0      ,0.0        ,204.082    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 123.457    ,204.082    ,0.0        ,400.0      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 59.172     ,0.0        ,400.0      ,0.0        ,0.0        ,0.0        ,0.0        ,82.645     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 100.0      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,27.701     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 277.778    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,277.778    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 14.793     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,30.864     ,0.0         ],
                [ 204.082    ,0.0        ,0.0        ,82.645     ,0.0        ,0.0        ,0.0        ,0.0        ,51.02      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 82.645     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,51.02      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,25.0        ],
                [ 204.082    ,0.0        ,0.0        ,0.0        ,27.701     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 82.645     ,0.0        ,0.0        ,0.0        ,0.0        ,277.778    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,82.645     ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 277.778    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,82.645     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0         ],
                [ 400.0      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,625.0      ,0.0        ,625.0       ],
                [ 123.457    ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,625.0      ,0.0        ,0.0        ,0.0         ],
                [ 100.0      ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,30.864     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,277.778     ],
                [ 30.864     ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,0.0        ,25.0       ,0.0        ,0.0        ,0.0        ,625.0      ,0.0        ,277.778    ,0.0         ],
                ])
    netbfe_std = cal_std(invv, delta_value=0.2)
    print_array('netbfe_std', netbfe_std)

    # optimal
    sij = np.array([
                [ 155.123     ,292.37      ,380.137     ,688.092     ,745.547     ,311.446     ,1211.053    ,416.532     ,752.001     ,361.081     ,373.612     ,331.513     ,258.844     ,362.869     ,536.06      ,365.715      ],
                [ 292.37      ,155.123     ,263.26      ,155.119     ,155.119     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.119     ,155.119     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 380.137     ,263.26      ,155.124     ,207.461     ,155.119     ,155.119     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 688.092     ,155.119     ,207.461     ,155.124     ,155.12      ,155.12      ,155.12      ,470.409     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 745.547     ,155.119     ,155.119     ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,506.413     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 311.446     ,155.119     ,155.119     ,155.12      ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.119     ,225.906     ,155.119     ,155.118     ,155.118     ,155.118     ,155.118      ],
                [ 1211.053    ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,1661.781    ,155.12       ],
                [ 416.532     ,155.12      ,155.12      ,470.409     ,155.12      ,155.12      ,155.12      ,155.124     ,556.911     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 752.001     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,556.911     ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,618.644      ],
                [ 361.081     ,155.118     ,155.119     ,155.12      ,506.413     ,155.119     ,155.12      ,155.12      ,155.12      ,155.123     ,155.119     ,155.118     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 373.612     ,155.119     ,155.12      ,155.12      ,155.12      ,225.906     ,155.12      ,155.12      ,155.12      ,155.119     ,155.124     ,416.724     ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 331.513     ,155.119     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,416.724     ,155.123     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 258.844     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.119     ,155.124     ,128.0       ,155.118     ,132.424      ],
                [ 362.869     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.119     ,128.0       ,155.124     ,155.118     ,155.118      ],
                [ 536.06      ,155.12      ,155.12      ,155.12      ,155.12      ,155.118     ,1661.781    ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.118     ,155.118     ,155.124     ,205.337      ],
                [ 365.715     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,618.644     ,155.119     ,155.12      ,155.119     ,132.424     ,155.118     ,205.337     ,155.124      ],
                ])
    nsofar = np.array([
                    [ 0.      ,2137    ,1115    ,1751    ,3474    ,1684    ,1356    ,2213    ,2921    ,1663    ,721     ,1908    ,1675    ,1016    ,1796    ,258      ],
                    [ 2137    ,0       ,884     ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 1115    ,884     ,0       ,1076    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 1751    ,0       ,1076    ,0       ,0       ,0       ,0       ,1143    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 3474    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,444     ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 1684    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,886     ,0       ,0       ,0       ,0       ,0        ],
                    [ 1356    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,5327    ,0        ],
                    [ 2213    ,0       ,0       ,1143    ,0       ,0       ,0       ,0       ,989     ,0       ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 2921    ,0       ,0       ,0       ,0       ,0       ,0       ,989     ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,598      ],
                    [ 1663    ,0       ,0       ,0       ,444     ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0        ],
                    [ 721     ,0       ,0       ,0       ,0       ,886     ,0       ,0       ,0       ,0       ,0       ,897     ,0       ,0       ,0       ,0        ],
                    [ 1908    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,897     ,0       ,0       ,0       ,0       ,0        ],
                    [ 1675    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,640     ,0       ,685      ],
                    [ 1016    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,640     ,0       ,0       ,0        ],
                    [ 1796    ,0       ,0       ,0       ,0       ,0       ,5327    ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,732      ],
                    [ 258     ,0       ,0       ,0       ,0       ,0       ,0       ,0       ,598     ,0       ,0       ,0       ,685     ,0       ,732     ,0        ],
                    ])

    # for Tyk2 run5 iter_2 A_optimize
    nmols = sij.shape[0]
    opts = {
        'nmols': nmols,
        'delta_N': 20011.0, # scaled
        'ste': np.array([0.2] + [None for _ in range(1, nmols)]),
        'abfe_cutoff': 3.125, # scaled
        'rbfe_cutoff': 3.125, # scaled
        'abfe_lambdas': 16,
        'rbfe_lambdas': 12,
        'N_scale': 6.25e-5
    }
    optimal_std = optimal_std(nsofar, sij, delta_value=0.2, **opts)
    print_array('optimal_std', optimal_std)

    # lomap
    sij = np.array([
                [ 155.123     ,292.37      ,380.137     ,688.092     ,745.547     ,311.446     ,1211.053    ,416.532     ,752.001     ,361.081     ,373.612     ,331.513     ,258.844     ,362.869     ,536.06      ,365.715      ],
                [ 292.37      ,155.123     ,263.26      ,155.119     ,155.119     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.119     ,155.119     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 380.137     ,263.26      ,155.124     ,207.461     ,155.119     ,155.119     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 688.092     ,155.119     ,207.461     ,155.124     ,155.12      ,155.12      ,155.12      ,470.409     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 745.547     ,155.119     ,155.119     ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,506.413     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 311.446     ,155.119     ,155.119     ,155.12      ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.119     ,225.906     ,155.119     ,155.118     ,155.118     ,155.118     ,155.118      ],
                [ 1211.053    ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,1661.781    ,155.12       ],
                [ 416.532     ,155.12      ,155.12      ,470.409     ,155.12      ,155.12      ,155.12      ,155.124     ,556.911     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 752.001     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,556.911     ,155.124     ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,618.644      ],
                [ 361.081     ,155.118     ,155.119     ,155.12      ,506.413     ,155.119     ,155.12      ,155.12      ,155.12      ,155.123     ,155.119     ,155.118     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 373.612     ,155.119     ,155.12      ,155.12      ,155.12      ,225.906     ,155.12      ,155.12      ,155.12      ,155.119     ,155.124     ,416.724     ,155.12      ,155.12      ,155.12      ,155.12       ],
                [ 331.513     ,155.119     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,416.724     ,155.123     ,155.119     ,155.119     ,155.12      ,155.119      ],
                [ 258.844     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.119     ,155.124     ,128.0       ,155.118     ,132.424      ],
                [ 362.869     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,155.12      ,155.119     ,155.12      ,155.119     ,128.0       ,155.124     ,155.118     ,155.118      ],
                [ 536.06      ,155.12      ,155.12      ,155.12      ,155.12      ,155.118     ,1661.781    ,155.12      ,155.12      ,155.12      ,155.12      ,155.12      ,155.118     ,155.118     ,155.124     ,205.337      ],
                [ 365.715     ,155.119     ,155.12      ,155.12      ,155.12      ,155.118     ,155.12      ,155.12      ,618.644     ,155.119     ,155.12      ,155.119     ,132.424     ,155.118     ,205.337     ,155.124      ],
                ])
    # here, nij corresponds to the next_nsofar of optimal section
    nij = np.array([
                [ 0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,26666666.7    ,26666666.7    ,26666666.7    ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0            ],
                [ 0.0           ,0.0           ,0.0           ,26666666.7    ,26666666.7    ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,26666666.7    ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0            ],
                [ 0.0           ,26666666.7    ,26666666.7    ,26666666.7    ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0            ],
                [ 0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,0.0           ,26666666.7     ],
                [ 0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,0.0            ],
                [ 26666666.7    ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7     ],
                [ 0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,0.0           ,26666666.7    ,0.0           ,26666666.7    ,0.0            ],
                ])
    lomap_std = lomap_std(nij, sij, delta_value=0.2)
    print_array('lomap_std', lomap_std)