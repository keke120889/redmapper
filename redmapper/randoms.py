"""Classes for generating redmapper randoms.
"""

import fitsio
import esutil

from .catalog import Catalog, Entry
from .galaxy import GalaxyCatalog, GalaxyCatalogMaker
from .cluster import ClusterCatalog

class GenerateRandoms(object):
    """
    Class to generate redmapper raw randoms using a redmapper volume limit mask.
    """

    def __init__(self, config, vlim_mask_or_file, redmapper_cat_or_file):
        """

        """

        self.config = config

        if isinstance(vlim_mask_or_file, VolumeLimitMask):
            self.vlim_mask = vlim_mask_or_file
        elif isinstance(vlim_mask_or_file, str):
            # This 0.2 is a dummy value
            self.vlim_mask = VolumeLimitMask(config, 0.2, vlimfile=self.vlim_mask_or_file)
        else:
            raise RuntimeError("vlim_mask_or_file must be a redmapper.VolumeLimitMask or a filename")

        if isinstance(redmapper_cat_or_file, ClusterCatalog):
            self.redmapper_cat = redmapper_cat_or_file
        elif isinstance(redmapper_cat_or_file, str):
            self.redmapper_cat = ClusterCatalog.from_fits_file(redmapper_file)
        else:
            raise RuntimeError("redmapper_cat_or_file must be a redmapper.GalaxyCatalog")

    def generate_randoms(self, nrandoms, outbase):
        """

        """

        min_gen = 10000
        max_gen = 1000000

        n_left = copy.copy(nrandoms)
        ctr = 0

        dtype = [('mem_match_id', 'i4'),
                 ('ra', 'f8'),
                 ('dec', 'f8'),
                 ('z', 'f4'),
                 ('lambda', 'f4'),
                 ('id_input', 'i4')]

        # randcat = Catalog(np.zeros(nrandoms, dtype=dtype))
        # randcat.mem_match_id = np.arange(nrand) + 1

        info_dict = {}
        maker = RandomCatalogMaker(outbase, info_dict)

        while (n_left > 0):
            n_gen = np.clip(n_left * 3, min_gen, max_gen)
            ra_rand, dec_rand = healsparse.makeUniformRandoms(self.vlim_mask.sparse_vlimmap, n_gen)

            zmax, fracgood = self.vlim_mask.calc_zmax(ra_rand, dec_rand, get_fracgood=True)

            r = np.random.uniform(size=n_gen)
            gd, = np.where(r < fracgood)

            if gd.size == 0:
                continue

            tempcat = Catalog(np.zeros(gd.size, dtype=dtype))
            tempcat.ra = ra_rand[gd]
            tempcat.dec = dec_rand[gd]
            tempcat.z = -1.0

            r = np.random.choice(np.arange(self.redmapper_cat.size), size=gd.size, replace=True)
            zz = self.redmapper_cat.z_lambda[r]
            ll = self.redmapper_cat.Lambda[r]
            ii = self.redmapper_cat.mem_match_id[r]

            # zctr counts the number of successfully placed randoms
            # while i counts index through tempcat, many of which will be rejected.
            zctr = 0
            for i in range(tempcat.size):
                if (zz[zctr] < zmax[i]):
                    # This is in a location that is within the volume limit.
                    tempcat.z[i] = zz[zctr]
                    tempcat.Lambda[i] = ll[zctr]
                    tempcat.id_input[i] = ii[zctr]
                    zctr += 1

            # Which of the tempcat were actually placed?
            gd, = np.where(tempcat.z > 0.0)
            n_good = gd.size

            if n_good == 0:
                continue

            if n_good > n_left:
                n_good = n_left

            tempcat = tempcat[gd]
            tempcat.mem_match_id = np.arange(ctr + 1, ctr + n_good + 1)

            maker.append_randoms(tempcat._ndarray[: n_good])

            ctr += n_good
            n_left -= n_good

        maker.finalize_catalog()


class RandomCatalog(GalaxyCatalog):
    """
    """

    @classmethod
    def from_randfile(cls, filename, nside=0, hpix=[], border=0.0):
        """
        """

        return super(RandomCatalog, cls).from_galfile(filename, nside=nside, hpix=hpix, border=border)

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=[], border=0.0, truth=False):
        raise NotImplementedError("Cannot call from_galfile on a RandomCatalog")

    @property
    def galcol(self):
        raise NotImplementedError("Cannot call galcol on a RandomCatalog")

    @property
    def galcol_err(self):
        raise NotImplementedError("Cannot call galcol_err on a RandomCatalog")

    @property
    def add_zred_fields(self):
        raise NotImplementedError("Cannot call add_zred_fields on a RandomCatalog")


class RandomCatalogMaker(GalaxyCatalogMaker):
    """
    """

    def __init__(self, outbase, info_dict, nside=32, maskfile=None, mask_mode=0, parallel=False):
        """
        """

        if 'LIM_REF' not in info_dict:
            info_dict['LIM_REF'] = 0.0
        if 'REF_IND' not in info_dict:
            info_dict['REF_IND'] = 0
        if 'AREA' not in info_dict:
            info_dict['AREA'] = 0.0
        if 'NMAG' not in info_dict:
            info_dict['NMAG'] = 0
        if 'MODE' not in info_dict:
            info_dict['MODE'] = 'NONE'

        super(RandomCatalogMaker, self).__init__(outbase, info_dict, nside=nside, maskfile=maskfile, mask_mode=mask_mode, parallel=parallel)

    def split_randoms(self, rands):
        """
        """

        if self.is_finalized:
            raise RuntimeError("Cannot split randoms for an already finalized catalog.")
        if os.path.isfile(self.filename):
            raise RuntimeError("Cannot split randoms when final file %s already exists." % (self.filename))

        self.append_randoms(rands)
        self.finalize_catalog()

    def append_randoms(self, rands):
        """
        """

        self.append_galaxies(rands)

    def _check_galaxies(self, rands):
        # These always come back true.
        return True

