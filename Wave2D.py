# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:40:27 2024

@author: stian
"""

import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y,t, c,  L = sp.symbols('x,y,t,c,L')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        
        x = y = np.linspace(0, self.L, N+1)
        
        self.xji, self.yij = np.meshgrid(x,y, indexing= 'ij', sparse = sparse)
        

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1,-2,1], [-1,0,1], (N+1,N+1),'lil')
        D[0, : 4] = 2,-5,4,-1
        D[-1,-4:] = -1,4,-5,2
        
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)
        

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.dx = self.L / self.N
        self.dt = self.w *self.dx /self.c
        
        u0 = sp.lambdify(x, y, self.ue.subs({L: self.L, c: self.c, t: 0}))
        self.unm1[:] = u0(self.xji, self.yij)
        plotdata = {0:self.unm1.copy()}
        self.un[:] = sp.lambdify(x, y, self.ue.subs({L: self.L, c: self.c, t: self.dt}))
        
        plotdata[1] = self.un.copy()
        
        return plotdata
        

    @property
    def dt(self):
        """Return the time step"""
        L = 1
        self.L = L
        dt = self.L / self.N
        return dt #eller Nt?
        
        

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        
        raise NotImplementedError

    def apply_bcs(self):
        
        raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.ST = store_data
        self.Nt = Nt
        self.unp1 =self.un =self.unm1 =np.zeros(N+1)
        
        

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1,-2,1], [-1,0,1], (N+1,N+1),'lil')
        D[0, : 4] = -2,2,0,0
        D[-1,-4:] = 0,0,2,-2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
    
    
    def apply_bcs(self):
        self.u = np.zeros(self.N+1)
        self.u[0:0] = self.u[-1:-1] = 0
        
        for n in range(1,self.N-1):
            for j in range (1, self.N-1):
                possition = u[n:j+1] - 2*u[n:j] + u[n:j-1]
                u[n+1:j] = 2*u[n:j] - (c*dt / dx)**2 * possition
                possition = 0
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2
    
test_convergence_wave2d()
def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05
    

def test_exact_wave2d():
    
    raise NotImplementedError