# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

import sys

from math import pi, cos, sin, acos, sqrt

import numpy as np

from ouluspin._fortran import fortran_utils as fu
from ouluspin import result_table


class SimpleGrid:
    """A class to construct a simple grid consisting of either one Cartesian
    unit vector or all three Cartesian unit vectors.

    Arguments
    ---------

    Optional arguments
    ------------------
    grid_type : int
        Has a value of either 0 (default), which means that the grid consists of
        all three Cartesian unit vectors, or 1 in which case a single Cartesian
        unit vector will be used.
    grid_vector : tuple of float
        A tuple defining the direction of the Cartesian unit vector if a grid
        consisting of a single Cartesian unit vector is requested (i.e.,
        grid_type = 1). The default is a unit vector along the z axis.

    Attributes
    ----------
    grid_type : int
        Has a value of either 0 (default), which means that the grid consists of
        all three Cartesian unit vectors, or 1 in which case a single Cartesian
        unit vector will be used.
    grid_vector : tuple of float
        A tuple defining the direction of the Cartesian unit vector if a grid
        consisting of a single Cartesian unit vector is requested (i.e.,
        grid_type = 1). The default is a unit vector along the z axis.
    vectors : array of float64
        A 3 x n_grid_points array containing the grid points as unit vectors.
    weights : vector of float64
        A vector containing the (normalized) weight of each grid point.
    n_grid_points : int
        The total number of grid points used.

    Private methods
    ---------------

    Public methods
    --------------
    data_table() : ResultTable
        Return a human-readable table of the grid.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """

    def data_table(self):
        """Return a table of the grid points as an instance of ResultTable.
        Each row contains the weight of one grid point, the Cartesian
        components of the corresponding vector and the squared norm of that
        vector, which is a check of the normalization of the grid.
        """
        rows = []
        for i in range(0,self.n_grid_points):
            x = self.vectors[i][0]
            y = self.vectors[i][1]
            z = self.vectors[i][2]

            rows.append([self.weights[i],x,y,z,x**2 + y**2 + z**2])

        summary = ["Number of grid points: {0}".format(self.n_grid_points)]

        return result_table.ResultTable(rows,
                                        column_headers=['Weight','X','Y','Z','Norm'],
                                        title="SIMPLE CARTESIAN GRID",
                                        summary=summary,
                                        formats=['.8f','.6f','.6f','.6f','.6f'],
                                        table_type='grid')

    
    def __repr__(self):
        """Return a human-readable summary of the grid."""
        return str(self.data_table())
    

    def __init__(self, grid_type=0, grid_vector=(0.0,0.0,1.0)):
        """Construct the grid."""
        self.grid_type = grid_type

        if self.grid_type == 0:
            self.n_grid_points = 3
            self.vectors = np.zeros((3,3), dtype=np.float64)

            self.vectors[0][0] = 1.0
            self.vectors[1][1] = 1.0
            self.vectors[2][2] = 1.0

            self.weights = np.array([1.0/3.0,1.0/3.0,1.0/3.0])
            
        elif self.grid_type == 1:
            if not len(grid_vector) == 3:
                print("ERROR in SimpleGrid.")
                print("ERROR: Inconsistent grid vector dimension.")
                print("Error termination.")
                sys.exit(1)

            self.n_grid_points = 1
            
            vector_norm = 0.0
            for i in range(0,3):
                vector_norm += grid_vector[i]**2
            vector_norm = sqrt(vector_norm)

            self.grid_vector = []
            for i in range(0,3):
                self.grid_vector.append(grid_vector[i] / vector_norm)

            self.vectors = np.zeros((1,3), dtype=np.float64)
            for i in range(0,3):
                self.vectors[0][i] = self.grid_vector[i]
                
            self.weights = np.array([1.0])

        else:
            print("ERROR in SimpleGrid.")
            print("ERROR: Uknown grid type: " + str(self.grid_type))
            print("Error termination.")
            sys.exit(1)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('SimpleGrid',test_name,
                                                       condition,print_output))

        grid = cls(grid_type=0)
        check('Cartesian grid vectors',
              np.allclose(grid.vectors,np.identity(3)))
        check('Cartesian grid weights sum to one',
              abs(np.sum(grid.weights) - 1.0) < 1.0e-12)

        grid = cls(grid_type=1,grid_vector=(1.0,1.0,0.0))
        check('single-vector grid is normalized',
              abs(np.linalg.norm(grid.vectors[0]) - 1.0) < 1.0e-12)
        check('single-vector grid direction',
              np.allclose(grid.vectors[0],[1.0/sqrt(2.0),1.0/sqrt(2.0),0.0]))
        check('data table renders', len(str(grid.data_table())) > 0)

        return debug_output.test_summary('SimpleGrid',result_list,print_output)



class LebedevLaikovGrid:
    """A class for the construction and storage of Lebedev--Laikov grids. The grid
    construction is carried out by a Fortran routine that interfaces with the
    original set of Fortran 77 routines.

    The grid construction is based on the Fortran 77 code published at CCL.NET
    and was downloaded on 17 January 2022. The routines were originally written in C
    by D. Laikov (Moscow State University) and translated to Fortran by C. W. van
    W\"ullen (Ruhr-Universitaet). All credits for the implementation should be
    addressed to them. Orignal citation for the grid construction itself is:

        V. I. Lebedev and D. N. Laikov.
        A quadrature formula for the sphere of the 131st algebraic order of accuracy
        Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477--481.

    The quality of the grid is determined by the single argument grid_quality.
    To see the number of grid points each value corresponds to, check the
    grid_quality_dict dictionary in the class constructor.

    Choice of the grid quality
    --------------------------
    The default, grid_quality = 17 (590 points), follows from a benchmark of
    the powder-integrated static magnetic properties of a strongly
    anisotropic Dy(III) complex, whose ground doublet is close to a pure
    |J,M> = |15/2,+-15/2> and whose magnetization is therefore about as
    sharply peaked over the sphere as that of a real system gets. The table
    lists the largest deviation of the chi*T product (T from 2 to 300 K, in
    a field of 0.1 T) and of the magnetization (T from 1.8 to 5 K, fields up
    to 7 T) from the values converged on a grid of 5810 points:

        grid_quality   points   rel. dev. of chi*T   rel. dev. of M
                   5       50              2.5e-11           1.1e-2
                   7       86              3.3e-14           1.3e-2
                  11      194              6.8e-14           2.1e-3
                  14      302              1.0e-13           1.0e-3
                  17      590              8.3e-14           3.0e-4
                  20     1202              2.0e-14           4.8e-5

    The susceptibility converges to machine precision at once: it is
    measured in a field small enough that the magnetization is linear in the
    field, so the powder average is an integral of a quadratic form over the
    sphere, and the grid integrates such an integrand exactly. A calculation
    of the susceptibility alone therefore needs no more than grid_quality =
    7.

    The magnetization is the slow one, and it converges erratically rather
    than monotonically: the saturated magnetization of an anisotropic system
    has a cusp along the easy axis, and a quadrature built to be exact for
    polynomials gains little from that exactness. For the magnetization the
    ZCW grid is the better choice, reaching a given accuracy at roughly an
    order of magnitude fewer points (see ZCWGrid).

    Arguments
    ---------

    Optional arguments
    ------------------
    grid_quality : int
        A number between 1 and 32 that determines the quality of the grid.
        The default is 17, i.e. 590 grid points; see above.

    Attributes
    ----------
    grid_quality : int
        A number between 1 and 32 that determines the quality of the grid.
    vectors : array of float64
        A n_grid_points x 3 array containing the grid points as unit vectors.
    weights : vector of float64
        A vector containing the (normalized) weight of each grid point.
    n_grid_points : int
        The total number of grid points used.

    Private methods
    ---------------

    Public methods
    --------------
    data_table() : ResultTable
        Return a human-readable table of the grid.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """
    def data_table(self):
        """Return a table of the grid points as an instance of ResultTable.
        Each row contains the weight of one grid point, the Cartesian
        components of the corresponding vector and the squared norm of that
        vector, which is a check of the normalization of the grid.
        """
        rows = []
        for i in range(0,self.n_grid_points):
            x = self.vectors[i][0]
            y = self.vectors[i][1]
            z = self.vectors[i][2]

            rows.append([self.weights[i],x,y,z,x**2 + y**2 + z**2])

        summary = ["Number of grid points: {0}".format(self.n_grid_points)]

        return result_table.ResultTable(rows,
                                        column_headers=['Weight','X','Y','Z','Norm'],
                                        title="LEBEDEV--LAIKOV GRID",
                                        summary=summary,
                                        formats=['.8f','.6f','.6f','.6f','.6f'],
                                        table_type='grid')

    
    def __repr__(self):
        """Return a human-readable summary of the grid."""
        return str(self.data_table())
    

    def __init__(self,grid_quality=17):
        """Construct the grid."""
        self.grid_quality = grid_quality

        grid_quality_dict = {
             1 :    6,
             2 :   14,
             3 :   26,
             4 :   38,
             5 :   50,
             6 :   74,
             7 :   86,
             8 :  110,
             9 :  146,
            10 :  170,
            11 :  194,
            12 :  230,
            13 :  266,
            14 :  302,
            15 :  350,
            16 :  434,
            17 :  590,
            18 :  770,
            19 :  974,
            20 : 1202,
            21 : 1454,
            22 : 1730,
            23 : 2030,
            24 : 2354,
            25 : 2702,
            26 : 3074,
            27 : 3470,
            28 : 3890,
            29 : 4334,
            30 : 4802,
            31 : 5294,
            32 : 5810,
        }

        if self.grid_quality not in grid_quality_dict.keys():
            print("ERROR in LebedevLaikovGrid.")
            print("ERROR: Uknown grid quality. The grid quality should be an integer")
            print("       between 1 and 32.")
            print("Error termination.")
            sys.exit(1)

        self.n_grid_points = grid_quality_dict[self.grid_quality]

        self.vectors, self.weights = fu.grid_utils.generate_lebedev_laikov_grid(self.n_grid_points)


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('LebedevLaikovGrid',test_name,
                                                       condition,print_output))

        default_grid = cls()
        check('the default grid quality is the recommended one',
              default_grid.grid_quality == 17
              and default_grid.n_grid_points == 590)

        grid = cls(3)
        check('number of grid points', grid.n_grid_points == 26)
        check('weights sum to one', abs(np.sum(grid.weights) - 1.0) < 1.0e-12)
        check('grid vectors are unit vectors',
              np.allclose(np.linalg.norm(grid.vectors,axis=1),1.0))
        # Exact spherical integrals: <n> = 0 and <n_z^2> = 1/3.
        check('first moment vanishes',
              np.allclose(np.dot(grid.weights,grid.vectors),0.0,atol=1.0e-12))
        check('second moment of n_z is 1/3',
              abs(np.dot(grid.weights,grid.vectors[:,2]**2) - 1.0/3.0) < 1.0e-12)
        check('data table renders', len(str(grid.data_table())) > 0)

        return debug_output.test_summary('LebedevLaikovGrid',result_list,print_output)



class ZCWGrid:
    """A class for the construction of a Zaremba--Conroy--Wolfsberg (ZCW) grid. The grid
    is used for obtaining a spherical distribution of field directions for the spherical
    integration of magnetic properties; namely, the magnetization.

    The term grid point here referes to a specific orientation defining a point on a
    unit sphere. Each grid point is defined either by two Euler angles or by a Cartesian
    vector of unit length. Both the angles and vectors will be stored as attributes.

    The grid is used as defined in Appendix 1 of

        M. Edén and M. H. Levitt. J. Magn. Reson. 1998, 132, 220--239.

    The grid is the same as used in the PHI program suite. The class constructor takes as
    input a single integer parameter defining the accuracy of the grid. The grid can be
    constructed for a full sphere, hemisphere or octant.

    Choice of the M parameter
    -------------------------
    The default, M = 5 (233 points), follows from the same benchmark of the
    static magnetic properties of a strongly anisotropic Dy(III) complex
    that is described in the class documentation of LebedevLaikovGrid. The
    largest deviations from the converged values are:

        M    points   rel. dev. of chi*T   rel. dev. of M
        1        34               1.7e-3           1.4e-3
        3        89               2.5e-4           2.1e-4
        4       144               9.6e-5           8.0e-5
        5       233               3.7e-5           3.1e-5
        7       610               5.3e-6           4.5e-6
        9      1597               7.8e-7           6.5e-7

    Both properties converge smoothly, as the square of the number of grid
    points, and neither is favoured over the other: the points are
    distributed uniformly over the sphere and the grid makes no assumption
    about the integrand. This makes the grid the better choice for the
    magnetization, and the better choice overall when both properties are
    calculated: at M = 5 the magnetization is more accurate than on a
    Lebedev--Laikov grid of 1202 points, at a fifth of the computational
    effort. For the susceptibility alone the Lebedev--Laikov grid is
    superior; see its class documentation.

    Arguments
    ---------

    Optional arguments
    ------------------
    M : int
        The M parameter defining the accuracy of the grid. The default is 5,
        i.e. 233 grid points; see above.
    integration_range : str
        How large a part of the sphere is integrated. The allowed values are 'sphere'
        (default), 'hemisphere' and 'octant'.

    Attributes
    ----------
    M : int
        The M parameter defining the accuracy of the grid.
    integration_range : str
        How large a part of the sphere is integrated. The allowed values are 'sphere'
        (default), 'hemisphere' and 'octant'.
    c : tuple of int
        A tuple of three integers which differ based on the integration range.
    n_grid_points : int
        The total number of points in the grid.
    g_M : int
        The g_M value.
    vectors : array of float64
        A n_grid_points x 3 array containing the grid points as unit vectors.
    angle_list : list of tuple of float
        The (phi, theta) ZCW angle pairs of the grid points.
    vector_list : list of array of float64
        The grid points as a list of unit vectors (the same points as in
        the vectors array).
    weights : vector of float64
        A vector containing the (normalized) weight of each grid point.

    Private methods
    ---------------
    __calculate_grid_points()
        Evaluate the grid points and store them as an attribute.
    __g(N) : int
        Calculate the g_N value as defined in eq. (58) in Edén and Lewitt.

    Public methods
    --------------
    data_table() : ResultTable
        Return a human-readable table of the grid.

    Class methods
    -------------
    run_tests(print_output=True) : boolean
        Initiate the class and run a set of internal tests. Return True if all
        tests passed.
    """
    def __calculate_grid_points(self):
        """Evaluate the grid points and store them as an attribute."""
        self.angle_list  = []
        self.vector_list = []

        g  = self.g_M
        N  = self.n_grid_points
        c1 = self.c[0]
        c2 = self.c[1]
        c3 = self.c[2]

        for i in range(0,self.n_grid_points):
            # alpha is the azimuthal angle and beta the polar angle. The
            # arccos construction of beta produces the correct uniform
            # distribution of points on the sphere with uniform weights.
            alpha = 2*pi / c3 * ((float(i*g) / N) % 1.0)
            beta  = acos(c1 * (c2*((float(i) / N) % 1.0) - 1.0))

            x = sin(beta)*cos(alpha)
            y = sin(beta)*sin(alpha)
            z = cos(beta)

            self.vectors[i][0] = x
            self.vectors[i][1] = y
            self.vectors[i][2] = z

    
    def __g(self,N):
        """Calculate the g_N value as defined in eq. (58) in Edén and Lewitt."""
        g_list = [8,13]
        if N < 2:
            return g_list[N]
        else:
            for R in range(2,N+1):
                g_list.append(g_list[R-1] + g_list[R-2])

            return g_list[N]


    def data_table(self):
        """Return a table of the grid points as an instance of ResultTable.
        Each row contains the weight of one grid point, the Cartesian
        components of the corresponding vector and the squared norm of that
        vector, which is a check of the normalization of the grid.
        """
        rows = []
        for i in range(0,self.n_grid_points):
            x = self.vectors[i][0]
            y = self.vectors[i][1]
            z = self.vectors[i][2]

            rows.append([self.weights[i],x,y,z,x**2 + y**2 + z**2])

        summary = ["Grid range:            {0}".format(self.integration_range),
                   "Number of grid points: {0}".format(self.n_grid_points),
                   "g:                     {0}".format(self.g_M)]

        return result_table.ResultTable(rows,
                                        column_headers=['Weight','X','Y','Z','Norm'],
                                        title="ZCW GRID",
                                        summary=summary,
                                        formats=['.8f','.6f','.6f','.6f','.6f'],
                                        table_type='grid')


    def __repr__(self):
        """Return a human-readable summary of the grid."""
        return str(self.data_table())
    

    def __init__(self, M=5, integration_range='sphere'):
        """Construct the grid points."""
        self.M = M
        self.integration_range = integration_range

        if self.integration_range == 'sphere':
            self.c = (1,2,1)
        elif self.integration_range == 'hemisphere':
            self.c = (-1,1,1)
        elif self.integration_range == 'octant':
            self.c = (-1,1,4)
        else:
            print("ERROR in ZCWGrid.")
            print("ERROR: Unkown integration range.")
            print("Error termination.")
            sys.exit(1)

        self.g_M           = self.__g(M)
        self.n_grid_points = self.__g(M+2)

        self.weights = np.zeros(self.n_grid_points, dtype=np.float64)
        w = 1.0 / self.n_grid_points
        for i in range(0,self.n_grid_points):
            self.weights[i] = w

        self.vectors = np.zeros((self.n_grid_points,3), dtype=np.float64)
        self.__calculate_grid_points()


    @classmethod
    def run_tests(cls, print_output=True):
        """Initiate the class and run a set of tests on the class constructor
        and the class methods. Return True if all tests passed and False
        otherwise.

        Optional arguments
        ------------------
        print_output : boolean
            Whether to print the outcome of each test. Default is True.
        """
        from ouluspin import _debug as debug_output

        result_list = []
        def check(test_name,condition):
            result_list.append(debug_output.test_check('ZCWGrid',test_name,
                                                       condition,print_output))

        default_grid = cls()
        check('the default M is the recommended one',
              default_grid.M == 5 and default_grid.n_grid_points == 233)

        grid = cls(8)
        check('number of grid points is a Fibonacci-type number',
              grid.n_grid_points == grid._ZCWGrid__g(10))
        check('weights sum to one', abs(np.sum(grid.weights) - 1.0) < 1.0e-12)
        check('grid vectors are unit vectors',
              np.allclose(np.linalg.norm(grid.vectors,axis=1),1.0))
        # A uniform spherical distribution: <n> = 0 and <n_a^2> = 1/3 for all
        # Cartesian components. The ZCW grid is quasi-random, so these hold
        # only approximately.
        check('first moment approximately vanishes',
              np.allclose(np.dot(grid.weights,grid.vectors),0.0,atol=5.0e-3))
        check('second moments are approximately 1/3',
              np.allclose(np.dot(grid.weights,grid.vectors**2),1.0/3.0,atol=5.0e-3))

        hemisphere = cls(6,integration_range='hemisphere')
        check('hemisphere grid stays in the upper hemisphere',
              np.all(hemisphere.vectors[:,2] >= -1.0e-12))
        check('data table renders', len(str(hemisphere.data_table())) > 0)

        return debug_output.test_summary('ZCWGrid',result_list,print_output)
