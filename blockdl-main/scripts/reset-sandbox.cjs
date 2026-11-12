const fs = require('fs');
const path = require('path');

const versionPath = path.resolve(__dirname, '../src/lib/sandbox-version.json');
const newVersion = Date.now();

const data = {
  resetVersion: newVersion,
  resetAt: new Date().toISOString()
};

fs.writeFileSync(versionPath, JSON.stringify(data, null, 2) + '\n', 'utf8');

console.log('\n======================================================');
console.log('✅ [VGG16 Progressive Sandbox] RESET AMBIENTE COMPLETATO!');
console.log(`   Versione reset generata: ${newVersion}`);
console.log('   All\'avvio o al prossimo refresh nel browser:');
console.log('   - Tutti i livelli (2..6) verranno ribloccati');
console.log('   - Tutte le modifiche salvate in cache verranno cancellate');
console.log('   - Il canvas si riaprirà pulito sul Livello 1 per gli studenti');
console.log('======================================================\n');

