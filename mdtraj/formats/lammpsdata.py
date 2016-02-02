##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2015 Stanford University and the Authors
#
# Authors: 
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
#
# Portions of this module originate from the ParmEd program, copyright (c) 2014
# Jason Swails, which is also distributed under the GNU Lesser General Public
# License
#
# Other portions of this code originate from the OpenMM molecular simulation
# toolkit, copyright (c) 2012 Stanford University and Peter Eastman. Those
# portions are distributed under the following terms:
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
##############################################################################
"""Load an md.Topology from CHARMM/XPLOR LMPD files
"""

# Written by Jason Swails <jason.swails@gmail.com> 9/8/2014
# This code was mostly stolen and stripped down from ParmEd

##############################################################################
# Imports
##############################################################################

from __future__ import print_function, division

from mdtraj.core import topology, element as elem
from mdtraj.formats import pdb
from mdtraj.utils.unit import unit_definitions as u
import numpy as np

__all__ = ['load_lmpd']

LMP_TYPE_COLUMNS = { 'molecular': ['atomid','moleculeid','atomtype','x','y','z'],
                     'full': ['atomid','moleculeid','atomtype','charge','x','y','z'],
                     'atomic': ['atomid','atomtype','x','y','z'],
                    }
#rest are integers
LMP_DOUBLE_TYPES = ['charge','x','y','z']

##############################################################################
# Functions
##############################################################################

class LMPDError(Exception):
    """ Raised upon problems parsing a LMPD file """
    pass

class _LMPDEOF(Exception):
    """ Raised when EOF is hit on the parsed LMPD """
    pass

def _convert(string, type, message):
    """Converts a string to the desired data type, making sure to raise LMPDError
    with the given message in the case of a failure.

    Parameters
    ----------
    string : str
        String to convert
    type : simple data type
        Either int, float, or str
    message : str
        Message to assign to the LMPDError if the conversion fails

    Returns
    -------
    The converted string to the desired datatype
    """
    try:
        return type(string)
    except ValueError:
        raise LMPDError('Could not convert %s [%s]' % (message, string))

def load_lmpd(fname, data_format='full'):
    """Load a Lammps data file from disk"

    Parameters
    ----------
    fname : str
        Path to the LMPD file on disk
    data_format: str
        Optionally specify format of lammps data file. 
        See http://lammps.sandia.gov/doc/read_data.html for further options. 
        Currently available: %s

    Returns
    -------
    top : md.Topology
        The resulting topology as an md.Topology object

    Notes
    -----
    Only the Masses Bonds and Atoms sections are read in, and all atoms are added to the
    same chain in the topology

    Raises
    ------
    LMPDError if any parsing errors occur

    Examples
    --------
    >>> topology = md.load_lmpd('mysystem.data')
    >>> # or
    >>> trajectory = md.load('trajectory.lammpstrj', top='system.data')
    """%",".join( LMP_TYPE_COLUMNS.keys() )

    top = topology.Topology()
    
    system_info = {}
    type_info = {}
    with open(fname, 'r') as f:
        line = f.readline()
        # first line is a comment
        comment = line
        # next line is blank
        line = f.readline()
        if line.strip(): raise LMPDError('Second line of lammps data file should be blank')

        # now number of each element is listed
        line = f.readline()
        while line.strip():
            #example: 100 atoms
            #example: 95 bonds
            parts = line.split()
            system_info[parts[1].lower()] = int(parts[0])
            line = f.readline()
        # the above loop includes blank line after the block

        # now number of each topology element is listed
        line = f.readline()
        while line.strip():
            #example: 5 atom types
            #example: 10 bond types
            parts = line.split()
            type_info[parts[1].lower()] = int(parts[0])
            line = f.readline()
        # the above loop includes blank line after the block

        # now read in box dimensions
        line = f.readline()
        box_dims = []
        while line.strip():
            # example : -0.5 0.5 xlo xhi
            parts = line.split()
            box_dims.append( float(parts[1])-float(parts[0]) )
            line = f.readline()
        # the above loop includes blank line after the block
        if len(box_dims)==2:
            box_dims.append(0)
        unitcell_lengths = np.array(box_dims)

        # Now start reading sections
        # Will always read atoms and masses
        line = f.readline()
        # now read to the end of the file
        while line:
            section = line.strip().lower()
            if not section in system_info.keys()+["masses"]:
                raise LMPDError("Section '%s' is not valid. Check header of data file"%section) 
            #skip blank line
            line = f.readline()
            if section == "masses": 
                nlines = type_info['atom']
                masses = np.zeros((nlines, 1))
            else:
                nlines = system_info[section]

            if section == "atoms":
                column_labels = LMP_TYPE_COLUMNS[data_format]
                integer_labels = [(label,labelidx) for labelidx,label in enumerate(column_labels) if not label in LMP_DOUBLE_TYPES]

                coordinates = np.zeros((nlines, 3),dtype=np.double)
                charges = np.zeros((nlines, 1),dtype=np.double)

                atom_type_dict = {}
                for label,labelidx in integer_labels:
                    atom_type_dict[label] = np.zeros(nlines,dtype=np.int)

                if 'charge' in column_labels:
                    charge_column = column_labels.index('charge')
                else:
                    charge_column = -1

            if section == "bonds":
                # bond_type, idx1, idx2
                bond_table = np.zeros((nlines, 3),dtype=np.int)

            for lineidx in range(nlines):
                line = f.readline().strip()
                line_parts = line.split()
                if section == "masses":
                    masses[lineidx] = float( line_parts[-1] )

                if section == "atoms":
                    # future, may need change to support 2 dimensions
                    if lineidx == 0 and not len(line_parts) == len(column_labels):
                        LMPDError("Number of columns in Atoms section does not match format specified (%s)"%data_format)
                    # last 3 parts always xyz
                    coordinates[lineidx,:] = np.array( line_parts[-3:], dtype=np.double)
                    if charge_column > 0 : charges[lineidx] = float( line_parts[charge_column] )
                    for label,colidx in integer_labels:
                        atom_type_dict[label][lineidx] = int(line_parts[colidx])

                if section == "bonds":
                    # skip first column, bond index
                    bond_table[lineidx,:] = np.array( line_parts[1:], dtype=np.int)
                     
            # skip blank line
            line = f.readline()
            # read hopefully the next header
            line = f.readline()
            
 
    prev_residue = (None, None, None)
    last_chain = None

    #these in all types
    atomids = atom_type_dict['atomid']
    atomtypes = atom_type_dict['atomtype']

    # leave this functionality in case it is desired to specify spliting system into named 'chains'
    segid = "system"
    resid = 1
    rname = resid

    #pdb.PDBTrajectoryFile._loadNameReplacementTables()

    natoms = system_info['atoms']
    for i in range(natoms):
        atomid = atomids[i]
        if atomid != i + 1:
            raise LMPDError('Nonsequential atom indices detected!')
        atomtype = atomtypes[i]
        mass = masses[atomtype-1]

        if 'moleculeid' in atom_type_dict:
            resid = atom_type_dict['moleculeid'][i]
        else:
            resid = i+1
        rname = str(resid)
        name = str(atomtype)
        charge = charges[i]

        if last_chain != segid:
            c = top.add_chain()
            last_chain = segid
        curr_residue = (resid, rname, segid)
        if prev_residue != curr_residue:
            prev_residue = curr_residue
            r = top.add_residue(rname, c, resid)
            r.segment_id = segid

        # Try to guess the element from the atom name for some of the common
        # ions using the names that CHARMM assigns to ions. If it's not one of
        # these 'weird' ion names, look up the element by mass. If the mass is
        # 0, assume a lone pair

        if mass == 0:
            element = elem.virtual
        else:
            element = elem.Element.getByMass(mass)

        # need special case for not real elements

        a = top.add_atom(name, element, r)
        a.charge = charge
        a.mass = mass

    # Add bonds to the topology
    atoms = list(top.atoms)
    nbonds = len(bond_table)

    for i in range(nbonds):
        top.add_bond(atoms[bond_table[i,1]-1], atoms[bond_table[i,2]-1])

    return top
