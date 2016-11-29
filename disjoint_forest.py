import numba  # most likely requires numba 0.29 to support recursion

"""
The pointer-tree structure used by `find_root` and `union` uses 1D (i,e, flat) indices for
the nodes of the graph structure. A negative value indicates the element in the matrix is a
root of a connected component; the magnitude of that value indicates the total number of
elements in that connected component. Otherwise a non-negative value is the flat-index that
points to another element in its connected component.

For example, the pointer-tree:
    [-3, 0, 0, -2, 3]
corresponds to a graph with two connected components. element-0 is the root of the component
containing {0, 1, 2}. element-3 is the root of the component containing {3, 4}.
"""


@numba.jit(nopython=True)
def find_root(x, pntr_tree):
    """ Returns the root node-ID of the connected component containing `x`

        Performs path compression. I.e. redirects all pointer values
        along the recursive path to point directly to the root node, compressing
        future root finding paths.

        Parameters
        ----------
        x : int
            A valid node-ID.
        pntr_tree : Sequence[int, ...]
            A pointer-tree, indicating the connected component membership of nodes in
            the graph.

        Returns
        -------
        int
            The root node-ID of the connected component `x` """
    if pntr_tree[x] < 0:  # x is the root of a connected component
        return x
    pntr_tree[x] = find_root(pntr_tree[x], pntr_tree)  # find the root that x points to, and update tree
    return pntr_tree[x]


@numba.jit(nopython=True)
def union(x, y, pntr_tree):
    """ Joins the connected components containing `x` and `y`, respectively.

        Performs union by rank: the root of the smaller component is pointed
        to the root of the larger one.

        Parameters
        ----------
        x : int
            A valid node-ID
        y : int
            A valid node-ID.
        pntr_tree : Sequence[int, ...]
            A pointer-tree, indicating the connected component membership of nodes in
            the graph."""
    r_x = find_root(x, pntr_tree)
    r_y = find_root(y, pntr_tree)

    if r_x != r_y:
        if pntr_tree[r_x] <= pntr_tree[r_y]:  # subgraph containing x is bigger (in magnitude!!)
            pntr_tree[r_x] += pntr_tree[r_y]  # add the smaller subgraph to the larger
            pntr_tree[r_y] = r_x  # point root of cluster y to x
        else:
            pntr_tree[r_y] += pntr_tree[r_x]
            pntr_tree[r_x] = r_y
    return None
