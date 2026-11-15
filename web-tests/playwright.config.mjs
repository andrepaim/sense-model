import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'.',testMatch:'*.spec.mjs',workers:1,timeout:45000,
 use:{baseURL:'http://127.0.0.1:18865',headless:true,viewport:{width:1440,height:1000}},
 webServer:{command:'../.venv/bin/python server_fixture.py',url:'http://127.0.0.1:18865',reuseExistingServer:false,timeout:30000},reporter:'list'});
