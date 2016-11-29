## Summary
[This notebook](./Labelling_Connected_Subgraphs.ipynb) implements a simple [two-pass algorithm for labeling the connected components of a graph](https://en.wikipedia.org/wiki/Connected-component_labeling#Two-pass) on a 2D lattice. Specifically, a 2D matrix of floats is specified along with a threshold value. Values beneath the threshold are mapped to 0 ('background'), otherwise the values are mapped to 1 ('foreground'). Assuming 8-connectivity, adjacent foreground elements are considered to be 'connected' to one another. This is performed using a union-find data structure, which leverages [*union by rank* and *path compression*](https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Disjoint-set_forests).

The 2D matrix and threshold value are passed to `first_pass`. This produces a pointer-tree: a 1D array of $N_{row} \times N_{col}$ integers. The elements of the pointer tree are sorted in conjunction with the matrix in row-major ordering. A negative value indicates the element in the matrix is a root of a connected component; the magnitude of that value indicates the total number of elements in that connected component. Otherwise a non-negative value is the flat-index that points to another element in its connected component.

For example, the pointer-tree: 
```python    
   [-3, 0, 0, -2, 3]
```
corresponds to a graph with two connected components. element-0 is the root of the component
containing `{0, 1, 2}`. element-3 is the root of the component containing `{3, 4}`.

`second_pass` then transforms the pointer tree such that all non-negative elements point directly to the root of a connected component. The 'background' elements are all grouped into a single '~~connected~~' component

## Example

![Example Image](./ConnectComponentExample.png)

```python
# An input image
>>> img = np.array([ [0, 1, 1, 0],
                     [0, 0, 1, 0],
                     [1, 0, 0, 0],
                     [1, 1, 1, 1]])

# The connected-component pointer tree 
# (reshaped so that it is easy to correspond with the matrix)
>>> first_pass(img, 0.99).reshape(4,4)  
array([[-8, -3,  1,  0],
       [ 0,  0,  1,  0],
       [-5,  0,  0,  0],
       [ 8,  8,  8,  8]], dtype=int64)
```

