# ModuleNotFoundError: No module named 'context_locals'
- Error:
    + File "D:\Projectspace\Virtual_Environment\xaiod_env\lib\site-packages\tool\__init__.py", line 11, in \<module\>
    + from context_locals import app, local
    + ModuleNotFoundError: No module named 'context_locals'
- Reason:
    + The package is written for Python 2, and uses implicit local imports. They were prohibited in Python 3.
- Solution:
    + Use the local package in: D:\Projectspace\Virtual_Environment\xaiod_env\Lib\site-packages\tool
    + Change "from context_locals import app, local" in \__init__.py to "from .context_locals import app, local"
- Suggestion:
    + Also change "from application import Application, WebApplication" in \__init__.py to "from .application import Application, WebApplication"

# ImportError: cannot import name 'Local' from 'werkzeug'
- Error:
    + File "D:\Projectspace\Virtual_Environment\xaiod_env\lib\site-packages\tool\context_locals.py", line 21, in \<module\>
    + from werkzeug import Local, LocalManager, LocalProxy
    + ImportError: cannot import name 'Local' from 'werkzeug'
- Reason:
    + If the version is 2.x or later, this error may occur because Local has been refactored.
- Solution: 
    + In modern Werkzeug versions, the Local class is available under werkzeug.local.
    + from werkzeug.local import Local

# ModuleNotFoundError: No module named 'tool'
- Currently, there are errors with "from tool import get_prediction, bbox_iou" in dclose.py
- Properly missing some files or wrong path
- If use "pip install tool", the "tool" installed may not be suitable as it is described as "A compact modular conf/web/console framework."
    + https://pypi.org/project/tool/
- As a result, a suggestion is to look back on YOLOX and its scripts.