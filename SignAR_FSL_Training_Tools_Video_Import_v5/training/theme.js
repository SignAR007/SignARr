(() => {
  const storageKey = 'signar-theme';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  const root = document.documentElement;
  let preference = null;
  let storageReadFailed = false;

  try {
    const savedTheme = window.localStorage.getItem(storageKey);
    if (savedTheme === 'light' || savedTheme === 'dark') preference = savedTheme;
  } catch (error) {
    storageReadFailed = true;
    console.warn('SignAR could not read the saved color mode.', error);
  }

  let currentTheme = preference || (systemTheme.matches ? 'dark' : 'light');

  function applyTheme(theme) {
    currentTheme = theme;
    root.dataset.theme = theme;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.content = theme === 'dark' ? '#11283f' : '#f1f6fb';
  }

  applyTheme(currentTheme);

  document.addEventListener('DOMContentLoaded', () => {
    const themeButton = document.getElementById('themeToggle');
    const toast = document.getElementById('toast');
    let toastTimer;

    if (!themeButton) {
      console.error('SignAR color mode control is missing.');
      return;
    }

    function announce(message) {
      if (!toast) {
        console.warn(message);
        return;
      }

      toast.textContent = message;
      toast.classList.add('show');
      window.clearTimeout(toastTimer);
      toastTimer = window.setTimeout(() => toast.classList.remove('show'), 3200);
    }

    function updateButton() {
      const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
      themeButton.textContent = nextTheme === 'dark' ? 'Dark mode' : 'Light mode';
      themeButton.setAttribute('aria-label', `Switch to ${nextTheme} mode`);
      themeButton.title = `Switch to ${nextTheme} mode`;
    }

    themeButton.addEventListener('click', () => {
      preference = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(preference);
      updateButton();

      try {
        window.localStorage.setItem(storageKey, preference);
      } catch (error) {
        console.warn('SignAR could not save the color mode preference.', error);
        announce('Theme changed for this visit, but your preference could not be saved.');
      }
    });

    const updateFromSystem = event => {
      if (preference !== null) return;
      applyTheme(event.matches ? 'dark' : 'light');
      updateButton();
    };

    if (typeof systemTheme.addEventListener === 'function') {
      systemTheme.addEventListener('change', updateFromSystem);
    } else {
      systemTheme.addListener(updateFromSystem);
    }

    window.addEventListener('storage', event => {
      if (event.key !== storageKey && event.key !== null) return;
      preference = event.newValue === 'light' || event.newValue === 'dark' ? event.newValue : null;
      applyTheme(preference || (systemTheme.matches ? 'dark' : 'light'));
      updateButton();
    });

    updateButton();
    if (storageReadFailed) announce('Using your system color mode; saved preferences are unavailable in this browser.');
  });
})();
