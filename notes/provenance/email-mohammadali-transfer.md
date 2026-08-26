Subject: SLAC → UTA data transfer: script + prioritized list (attached)

Hi Mohammadali,

We tracked down the full production behind Umar's ntuples via the PanDA task
record: the 17 files we have at UTA are only ~0.6% of a 44.7M-event ttbar
production, and there's also a small VBF H→invisible sample from the same
dumper. Since you have SLAC access, could you pull more of it to our GPU
server? The attached script (slac_pull.sh) should replace your
stage-on-the-jump-node workflow with a single direct stream.

HOW THE SCRIPT WORKS

It runs ON cn-1e1901 (our GPU server), not at SLAC. It's rsync over an ssh
tunnel: ssh's ProxyJump opens a connection to an interior S3DF node
*through* the login node — the login node just forwards encrypted bytes,
nothing is written there — and rsync streams the files over that channel
straight into /storage. So the data goes

  interior node filesystem → (tunnel through s3dflogin) → cn-1e1901:/storage

in one hop from your point of view: no staging copy at SLAC, nothing in any
home directory. It also uses ssh connection-sharing (ControlMaster), so you
authenticate ONCE (`./slac_pull.sh auth`, one password/Duo prompt) and every
subsequent command reuses that live connection for 12 hours with no further
prompts. Transfers are resumable: if the link drops, rerun the same command
— completed files are skipped and partial files continue where they left
off (and the script itself retries a few times on its own).

SETUP (one time, ~1 minute)

Edit the config block at the top of the script:
  SLAC_USER=<your SLAC username>
  INNER=<the interactive node you normally use>   # e.g. sdfiana or an
                                                  # sdf-cpu node — the data
                                                  # isn't visible on the
                                                  # login node, so this is
                                                  # required
  JUMP=s3dflogin.slac.stanford.edu                # change if you use another

We've verified cn-1e1901 can reach s3dflogin on port 22, so no SLAC-side
setup should be needed.

USAGE

  ./slac_pull.sh auth                            # one Duo prompt
  ./slac_pull.sh list  <path_to_vbf_container>   # saves an ls manifest
  ./slac_pull.sh pull  <path_to_vbf_container>  vbf_hinv
  ./slac_pull.sh pull  <path_to_ttbar_container> ttbar --max-files 150

Options: --dry-run (preview), --bwlimit <KB/s> (throttle if the link is
shared), --files <listfile> (explicit selection).

You know the S3DF layout better than we do — the two containers to locate
are:

  user.bbullard.mc21_14TeV.600026.PhH7EG_NNPDF3_AZNLO_VBFH125_ZZ4nu_MET75.ntuple.e8481_s4290_r15700.20260608_ntuple.root
  user.bbullard.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.ntuple.e8481_s4446_r16176.20260604_ntuple.root

(wherever Umar's group keeps them under /sdf/... — if you can't find them,
Umar will know.)

WHAT TO COPY, IN ORDER

P1 — the VBF H→invisible slice first (small): all 10 files of the 600026
  container (~50k events, tens of GB)
  → /storage/mxg1065/superhjd/vbf_hinv/

P2 — ttbar extension (~1 TB): any ~150 files of the 601229 container that
  we DON'T already hold — we have file numbers 000011–000020, 000023,
  000025, 000028–000032. Contiguous numbers preferred.
  → /storage/mxg1065/superhjd/ttbar/
  (~10× our statistics; /storage has 6.5 TB free, so 1 TB is comfortable.)

P3 — later, once P1+P2 look good: keep extending ttbar in ~0.5 TB chunks,
  up to ~2.5 TB total for now.

P4 — small extras if you can find them: the SuperHJD package source (repo
  + tag, or a tarball), and the `list` manifests of both containers before
  you pull (so we can verify counts against the production record).

Separately — for the dataset section of our notes: which
generator/tune/PDF and simulation chain produced the ttbar and HH→bbττ
samples behind your Calo ntuples? Just the ATLAS dataset names are enough
if that's easiest.

Thanks!
Amir
