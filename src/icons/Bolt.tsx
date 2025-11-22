import { Zap as LucideZap, LucideProps } from 'lucide-react';

const Bolt = ({ className, ...props }: LucideProps) => {
  return <LucideZap className={className} {...props} />;
};

export default Bolt;