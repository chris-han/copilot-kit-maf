import { Zap as LucideZap, LucideProps } from 'lucide-react';

const PlugIn = ({ className, ...props }: LucideProps) => {
  return <LucideZap className={className} {...props} />;
};

export default PlugIn;