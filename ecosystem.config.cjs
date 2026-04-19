module.exports = {
  apps: [
    {
      name: 'bookdialog',
      script: 'python3',
      args: '-m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info',
      cwd: '/home/user/bookdialog',
      env: { PYTHONUNBUFFERED: '1' },
      watch: false,
      instances: 1,
      exec_mode: 'fork',
      restart_delay: 3000,
      max_restarts: 5
    }
  ]
}
